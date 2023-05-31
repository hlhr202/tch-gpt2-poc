use std::{borrow::BorrowMut, collections::HashMap};

use tch::{
    nn::{EmbeddingConfig, Linear, Module},
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
        let c_fc_weight = tensor_map
            .get(&format!("{}.c_fc.weight", path))
            .unwrap_or_else(|| panic!("failed to get {}.c_fc.weight", path));
        let c_fc_bias = tensor_map
            .get(&format!("{}.c_fc.bias", path))
            .unwrap_or_else(|| panic!("failed to get {}.c_fc.bias", path));
        let c_proj_weight = tensor_map
            .get(&format!("{}.c_proj.weight", path))
            .unwrap_or_else(|| panic!("failed to get {}.c_proj.weight", path));
        let c_proj_bias = tensor_map
            .get(&format!("{}.c_proj.bias", path))
            .unwrap_or_else(|| panic!("failed to get {}.c_proj.bias", path));

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
        let activation = &self.activation.0;
        let h = activation(&self.c_fc.forward(xs));
        self.c_proj.forward(&h)
    }
}

#[derive(Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn from_path(path: &str, tensor_map: &HashMap<String, &Tensor>, eps: Option<f64>) -> Self {
        let weight = tensor_map
            .get(&format!("{}.weight", path))
            .unwrap_or_else(|| panic!("failed to get {}.weight", path));
        let bias = tensor_map
            .get(&format!("{}.bias", path))
            .unwrap_or_else(|| panic!("failed to get {}.bias", path));

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
            tensor_map,
            config.layer_norm_epsilon,
        );
        let attn = Attention::from_path(&format!("{}.attn", path), tensor_map, config, true);
        let ln_2 = LayerNorm::from_path(
            &format!("{}.ln_2", path),
            tensor_map,
            config.layer_norm_epsilon,
        );
        let mlp = Mlp::from_path(&format!("{}.mlp", path), tensor_map);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        layer_past: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let (output, present) =
            self.attn
                .forward(&self.ln_1.forward(x), layer_past, attention_mask);
        let x = x + output;
        let m = self.mlp.forward(&self.ln_2.forward(&x));
        let x = x + m;
        (x, present)
    }
}

#[derive(Debug)]
pub struct Embedding {
    pub ws: Tensor,
    pub config: EmbeddingConfig,
}

impl Module for Embedding {
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
    pub wte: Embedding,
    pub wpe: Embedding,
    pub h: Vec<Block>,
    pub ln_f: LayerNorm,
    pub output_past: bool,
    pub output_hidden_states: bool,
    // output_attentions: bool,
}

impl GPT2Model {
    fn new(tensor_map: &HashMap<String, &Tensor>, config: &Config) -> Self {
        let wte = Embedding {
            ws: tensor_map
                .get("wte.weight")
                .unwrap_or_else(|| panic!("failed to get wte.weight"))
                .shallow_clone(),
            config: Default::default(),
        };
        let wpe = Embedding {
            ws: tensor_map
                .get("wpe.weight")
                .unwrap_or_else(|| panic!("failed to get wpe.weight"))
                .shallow_clone(),
            config: Default::default(),
        };
        let h = (0..config.n_layer)
            .map(|i| Block::from_path(&format!("h.{}", i), tensor_map, config))
            .collect();
        let ln_f = LayerNorm::from_path("ln_f", tensor_map, config.layer_norm_epsilon);

        Self {
            wte,
            wpe,
            h,
            ln_f,
            output_past: true,
            output_hidden_states: false,
            // output_attentions: false
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        layer_past: Option<&Vec<Tensor>>,
        attention_mask: Option<&Tensor>,
    ) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let (past, past_length) = match layer_past {
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

        let input_shape = input_ids.size();
        let input_seq_length = input_shape[1];
        // let input_batch_size = input_shape[0];

        let position_ids = match position_ids {
            Some(value) => value.copy(),
            None => Tensor::arange_start(
                past_length,
                input_seq_length + past_length,
                (Kind::Int64, input_ids.device()),
            )
            .unsqueeze(0),
        };

        let inputs_embeds = self.wte.forward(input_ids);
        let position_embeds = self.wpe.forward(&position_ids);

        let token_type_embeds = match token_type_ids {
            Some(value) => self.wte.forward(value),
            None => Tensor::zeros_like(&position_embeds),
        };

        let attention_mask: Option<Tensor> = attention_mask.map(|value| {
            let attention_mask = value
                .view((inputs_embeds.size()[0], -1))
                .unsqueeze(1)
                .unsqueeze(2)
                .to_kind(inputs_embeds.kind());

            let attention_mask: Tensor = (1.0 - attention_mask) * (-10000.0);
            attention_mask.to_kind(inputs_embeds.kind())
        });

        let mut hidden_state = inputs_embeds + position_embeds + token_type_embeds;

        let mut all_presents: Option<Vec<Tensor>> =
            if self.output_past { Some(vec![]) } else { None };
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        // let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
        //     Some(vec![])
        // } else {
        //     None
        // };

        let layer_iter = self.h.iter().zip(past);
        for layer_values in layer_iter {
            let (layer, past) = layer_values;
            let temp = layer.forward(&hidden_state, past.as_ref(), attention_mask.as_ref());
            hidden_state = temp.0;
            if let Some(presents) = all_presents.borrow_mut() {
                presents.push(temp.1);
            };
            // if let Some(attentions) = all_attentions.borrow_mut() {
            //     attentions.push(temp.2.unwrap());
            // };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().copy());
            };
        }

        hidden_state = self.ln_f.forward(&hidden_state);

        (hidden_state, all_presents, all_hidden_states)
    }
}

#[derive(Debug)]
pub struct GPT2LMHead {
    decoder: Linear,
}

impl GPT2LMHead {
    fn new(model_embeddings_weights: &Tensor) -> Self {
        let decoder = Linear {
            ws: model_embeddings_weights.shallow_clone(),
            bs: None,
        };

        Self { decoder }
    }
}

impl Module for GPT2LMHead {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        self.decoder.forward(hidden_states)
    }
}

#[derive(Debug)]
pub struct GPT2LMHeadModel {
    pub transformer: GPT2Model,
    pub lm_head: GPT2LMHead,
}

impl GPT2LMHeadModel {
    pub fn new(tensor_map: &HashMap<String, &Tensor>, config: &Config) -> Self {
        let transformer = GPT2Model::new(tensor_map, config);
        let lm_head = GPT2LMHead::new(&transformer.wte.ws);

        Self {
            transformer,
            lm_head,
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        layer_past: Option<&Vec<Tensor>>,
        attention_mask: Option<&Tensor>,
    ) -> (Tensor, Option<Vec<Tensor>>) {
        let (hidden_states, presents, _) = self.transformer.forward(
            input_ids,
            position_ids,
            token_type_ids,
            layer_past,
            attention_mask,
        );
        let lm_logits = self.lm_head.forward(&hidden_states);
        (lm_logits, presents)
    }
}
