use std::collections::HashMap;

use tch::{
    nn::{Embedding, EmbeddingConfig, Module},
    Kind, Tensor,
};

use crate::{
    attention::{Attention, Conv1D},
    config::Config,
};

pub struct TensorFunction(Box<fn(&Tensor) -> Tensor>);

impl std::fmt::Debug for TensorFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "TensorFunction")
    }
}

#[derive(Debug)]
struct Mlp {
    c_fc: Conv1D,
    c_proj: Conv1D,
    activation: TensorFunction,
}

impl Mlp {
    pub fn from_path(path: &str, tensor_map: &HashMap<String, &Tensor>) -> Self {
        let c_fc_weight = tensor_map.get(&format!("{}.c_fc.weight", path)).unwrap();
        let c_fc_bias = tensor_map.get(&format!("{}.c_fc.bias", path)).unwrap();
        let c_proj_weight = tensor_map.get(&format!("{}.c_proj.weight", path)).unwrap();
        let c_proj_bias = tensor_map.get(&format!("{}.c_proj.bias", path)).unwrap();

        let c_fc = Conv1D {
            weight: c_fc_weight.shallow_clone(),
            bias: c_fc_bias.shallow_clone(),
        };

        let c_proj = Conv1D {
            weight: c_proj_weight.shallow_clone(),
            bias: c_proj_bias.shallow_clone(),
        };

        Self {
            c_fc,
            c_proj,
            activation: TensorFunction(Box::new(|xs| xs.gelu("none"))),
        }
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let h = (self.activation.0)(&xs.apply(&self.c_fc));
        h.apply(&self.c_proj)
    }
}

#[derive(Debug)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn from_path(path: &str, tensor_map: &HashMap<String, &Tensor>, eps: Option<f64>) -> Self {
        let weight = tensor_map.get(&format!("{}.weight", path)).unwrap();
        let bias = tensor_map.get(&format!("{}.bias", path)).unwrap();

        Self {
            weight: weight.shallow_clone(),
            bias: bias.shallow_clone(),
            eps: eps.unwrap_or(1e-12),
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let u = xs.mean_dim(-1, true, Kind::Float);
        let s = (xs - &u).pow_(2).mean_dim(-1, true, Kind::Float);
        let x = (xs - &u) / (&s + self.eps).sqrt();
        x * &self.weight + &self.bias
    }
}

#[derive(Debug)]
pub struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn from_path(path: &str, tensor_map: &HashMap<String, &Tensor>, config: &Config) -> Self {
        let ln_1 = LayerNorm::from_path(
            &format!("{}.ln_1", path),
            &tensor_map,
            config.layer_norm_epsilon,
        );
        let attn = Attention::from_path(&format!("{}.attn", path), &tensor_map, config, true);
        let ln_2 = LayerNorm::from_path(
            &format!("{}.ln_2", path),
            &tensor_map,
            config.layer_norm_epsilon,
        );
        let mlp = Mlp::from_path(&format!("{}.mlp", path), &tensor_map);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    fn forward_t(
        &self,
        x: &Tensor,
        layer_past: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let (output, present) =
            self.attn
                .forward_t(&self.ln_1.forward(x), layer_past, attention_mask);
        let x = x + output;
        let m = self.mlp.forward(&self.ln_2.forward(&x));
        let x = x + m;
        (x, present)
    }
}

#[derive(Debug)]
pub struct MyEmbedding {
    pub ws: Tensor,
    pub config: EmbeddingConfig,
}

impl Module for MyEmbedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::embedding(
            &self.ws,
            xs,
            self.config.padding_idx,
            self.config.scale_grad_by_freq,
            self.config.sparse,
        )
    }
}

#[derive(Debug)]
pub struct GPT2Model {
    wte: MyEmbedding,
    wpe: MyEmbedding,
    h: Vec<Block>,
    ln_f: LayerNorm,
}

impl GPT2Model {
    fn new(tensor_map: &HashMap<String, &Tensor>, config: &Config) -> Self {
        let wte = MyEmbedding {
            ws: tensor_map.get("wte.weight").unwrap().shallow_clone(),
            config: Default::default(),
        };
        let wpe = MyEmbedding {
            ws: tensor_map.get("wpe.weight").unwrap().shallow_clone(),
            config: Default::default(),
        };
        let h = (0..config.n_layer)
            .map(|i| Block::from_path(&format!("h.{}", i), tensor_map, config))
            .collect();
        let ln_f = LayerNorm::from_path("ln_f", tensor_map, config.layer_norm_epsilon);

        Self { wte, wpe, h, ln_f }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        layer_past: Option<&Vec<Tensor>>,
    ) {
        let (layer_past, layer_past_length) = match layer_past {
            Some(value) => {
                assert_eq!(
                    value.len(),
                    self.h.len(),
                    "Past activations vector must be of length equal to the number of layers"
                );
                (
                    value
                        .iter()
                        .map(|v| Some(v.copy()))
                        .collect::<Vec<Option<Tensor>>>(),
                    value[0].size()[3],
                )
            }
            None => {
                let mut out = Vec::with_capacity(self.h.len());
                out.resize_with(self.h.len(), || None::<Tensor>);
                (out, 0)
            }
        };
    }
}
