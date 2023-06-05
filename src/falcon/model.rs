#![allow(clippy::too_many_arguments)]
use std::{
    borrow::{Borrow, BorrowMut},
    vec,
};

use crate::config::RWConfig;
use tch::{
    nn::{
        embedding, layer_norm, EmbeddingConfig, Init, LayerNorm, LayerNormConfig, LinearConfig,
        Module, Path,
    },
    Device, IndexOp, Kind, NewAxis, Tensor,
};

fn dropout_add(x: &Tensor, residual: &Tensor, p: f64, train: bool) -> Tensor {
    let out = x.dropout(p, train);
    residual + out
}

fn make_causal_mask(
    input_ids_shape: (i64, i64),
    device: Device,
    past_key_values_length: i64,
) -> Tensor {
    let (batch_size, target_length) = input_ids_shape;
    let mask = Tensor::zeros(
        [target_length, target_length + past_key_values_length],
        (Kind::Bool, device),
    );

    let seq_ids = Tensor::arange(target_length, (Kind::Int64, device));

    mask.i((.., past_key_values_length..)).copy_(
        &seq_ids
            .i((.., NewAxis))
            .lt_tensor(&seq_ids.i((NewAxis, ..))),
    );

    if past_key_values_length > 0 {
        let mask_first_part = Tensor::zeros(
            [target_length, past_key_values_length],
            (Kind::Bool, device),
        );
        mask.i((.., ..past_key_values_length))
            .copy_(&mask_first_part);
    }

    mask.i((NewAxis, NewAxis, .., ..)).expand(
        [
            batch_size,
            1,
            target_length,
            target_length + past_key_values_length,
        ],
        false,
    )
}

fn expand_mask(mask: &Tensor, tgt_length: &Option<i64>) -> Tensor {
    let (batch_size, src_length) = mask.size2().unwrap();
    let tgt_length = tgt_length.unwrap_or(src_length);

    let expanded_mask = mask.i((.., NewAxis, NewAxis, ..)).to_kind(Kind::Bool).neg();

    expanded_mask.expand([batch_size, 1, tgt_length, tgt_length], false)
}

#[derive(Debug)]
pub struct Dropout {
    pub p: f64,
    pub train: bool,
}

impl Dropout {
    pub fn new(p: f64, train: bool) -> Self {
        Dropout { p, train }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.dropout(self.p, self.train)
    }
}

#[derive(Debug)]
pub struct Linear {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
}

/// Creates a new linear layer.
pub fn linear<'a, T: Borrow<Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
) -> Linear {
    let vs = vs.borrow();
    let bs = if c.bias {
        let bs_init = c.bs_init.unwrap_or_else(|| {
            let bound = 1.0 / (in_dim as f64).sqrt();
            Init::Uniform {
                lo: -bound,
                up: bound,
            }
        });
        Some(vs.var("bias", &[out_dim], bs_init))
    } else {
        None
    };

    Linear {
        ws: vs.var("weight", &[out_dim, in_dim], c.ws_init),
        bs,
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // transposed
        let ret = input.mm(&self.ws.t_copy());
        if let Some(bias) = &self.bs {
            ret + bias
        } else {
            ret
        }
    }
}

#[derive(Debug)]
pub struct RoteryEmbedding {
    inv_freq: Tensor,
    head_dim: i64,
    batch_size_cache: Option<i64>,
    cos_cached: Option<Tensor>,
    sin_cached: Option<Tensor>,
}

// dont know if this is correct
impl RoteryEmbedding {
    fn new(head_dim: i64, device: Device) -> Self {
        let base = 10000;
        let steps =
            Tensor::arange_start_step(0, head_dim, 2, (Kind::Float, device)) / head_dim as f64;

        let base = Tensor::from_slice(&vec![base; steps.size()[0] as usize]);

        let inv_freq = 1.0 / base.pow(&steps);

        RoteryEmbedding {
            inv_freq,
            head_dim,
            batch_size_cache: None,
            cos_cached: None,
            sin_cached: None,
        }
    }

    fn rotate_half(x: &Tensor) -> Tensor {
        // another approach
        // let x1 = x.slice(3, 0, x.size()[3], 2);
        // let x2 = x.slice(3, 1, x.size()[3], 2);
        // Tensor::stack(&[-x2, x1], -1).flatten(-2, -1)

        let x1 = x.narrow(-1, 0, x.size().last().unwrap() / 2);
        let x2 = x.narrow(
            -1,
            x.size().last().unwrap() / 2,
            x.size().last().unwrap() / 2,
        );
        Tensor::cat(&[&(-&x2), &x1], (x1.dim() - 1) as i64)
    }

    fn cos_sin(&mut self, seq_len: i64, device: Device, dtype: Kind) -> (Tensor, Tensor) {
        if self.cos_cached.is_none() || self.sin_cached.is_none() {
            let t = Tensor::arange(seq_len, (Kind::Float, device));
            // einsum("i,j->ij", t, self.inv_freq)
            let freqs = t.unsqueeze(1) * self.inv_freq.unsqueeze(0);
            let emb = Tensor::cat(&[freqs.shallow_clone(), freqs], -1).to_kind(dtype);

            let cos_cached = emb
                .cos()
                .view((1, -1, *emb.size().last().unwrap()))
                .to_kind(dtype);
            let sin_cached = emb
                .sin()
                .view((1, -1, *emb.size().last().unwrap()))
                .to_kind(dtype);
            self.cos_cached = Some(cos_cached);
            self.sin_cached = Some(sin_cached);
        }

        (
            self.cos_cached.as_ref().unwrap().shallow_clone(),
            self.sin_cached.as_ref().unwrap().shallow_clone(),
        )
    }

    fn forward(&mut self, q: &Tensor, k: &Tensor) -> (Tensor, Tensor) {
        let seq_len = q.size()[1];
        let (cos, sin) = self.cos_sin(seq_len, q.device(), Kind::BFloat16);
        (
            q * &cos + RoteryEmbedding::rotate_half(q) * &sin,
            k * &cos + RoteryEmbedding::rotate_half(k) * &sin,
        )
    }
}

#[derive(Debug)]
struct Attention {
    hidden_size: i64,
    num_heads: i64,
    head_dim: i64,
    multi_query: bool,
    maybe_rotary: Option<RoteryEmbedding>,
    query_key_value: Linear,
    num_kv: i64,
    dense: Linear,
    inv_norm_factor: f64,
    attention_dropout: Dropout,
}

impl Attention {
    pub fn new(path: &Path, config: &RWConfig) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.n_head;
        let head_dim = hidden_size / num_heads;
        // let split_size = hidden_size;
        let hidden_dropout = config.hidden_dropout;

        assert!(
            head_dim * num_heads == hidden_size,
            "hidden_size must be divisible by num_heads"
        );

        let maybe_rotary = if config.is_rotary() {
            Some(RoteryEmbedding::new(head_dim, config.get_device()))
        } else {
            // (q, k) -> (q, k)
            None
        };

        let inv_norm_factor = 1.0 / (head_dim as f64).sqrt();
        // let beta = inv_norm_factor;

        let out_dim = if !config.multi_query {
            3 * hidden_size
        } else {
            hidden_size + 2 * head_dim
        };

        let linear_config = LinearConfig {
            bias: config.bias,
            ..Default::default()
        };

        let query_key_value = linear(
            path / "query_key_value",
            hidden_size,
            out_dim,
            linear_config,
        );

        let dense = linear(path / "dense", hidden_size, hidden_size, linear_config);

        let attention_dropout = Dropout::new(hidden_dropout, false);

        let num_kv = if !config.multi_query {
            config.n_head
        } else {
            1
        };

        Attention {
            hidden_size,
            num_heads,
            head_dim,
            multi_query: config.multi_query,
            maybe_rotary,
            query_key_value,
            num_kv,
            dense,
            inv_norm_factor,
            attention_dropout,
        }
    }

    fn split_heads(&self, fused_qkv: &Tensor) -> (Tensor, Tensor, Tensor) {
        let batch_size = fused_qkv.size()[0];
        let seq_length = fused_qkv.size()[1];
        // let three_times_hidden_size = fused_qkv.size()[2];

        if !self.multi_query {
            let fused_qkv =
                fused_qkv.view([batch_size, seq_length, self.num_heads, 3, self.head_dim]);
            let fused_q = fused_qkv.index_select(-2, &Tensor::from_slice(&[0]));
            let fused_k = fused_qkv.index_select(-2, &Tensor::from_slice(&[1]));
            let fused_v = fused_qkv.index_select(-2, &Tensor::from_slice(&[2]));
            (fused_q, fused_k, fused_v)
        } else {
            let fused_qkv =
                fused_qkv.view([batch_size, seq_length, self.num_heads + 2, self.head_dim]);
            let fused_q = fused_qkv.narrow(-2, 0, self.num_heads);
            let fused_k = fused_qkv.narrow(-2, self.num_heads, 1);
            let fused_v = fused_qkv.index_select(-2, &Tensor::from_slice(&[-1]));
            (fused_q, fused_k, fused_v)
        }
    }

    fn merge_heads(&self, x: &Tensor) -> Tensor {
        let batch_size_and_num_heads = x.size()[0];
        let seq_length = x.size()[1];
        let batch_size = batch_size_and_num_heads / self.num_heads;
        let x = x
            .view((batch_size, self.num_heads, seq_length, self.head_dim))
            .permute([0, 2, 1, 3]);
        x.reshape([batch_size, seq_length, self.num_heads * self.head_dim])
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        alibi: &Option<Tensor>,
        attention_mask: &Tensor,
        layer_past: &Option<(Tensor, Tensor)>,
        head_mask: &Option<Tensor>,
        use_cache: bool,
        output_attentions: bool,
    ) -> (Tensor, Option<(Tensor, Tensor)>) {
        let fused_qkv = self.query_key_value.forward(hidden_states);
        let (query_layer, key_layer, value_layer) = self.split_heads(&fused_qkv);
        let batch_size = query_layer.size()[0];
        let q_length = query_layer.size()[1];
        let query_layer = query_layer.transpose(1, 2).reshape([
            batch_size * self.num_heads,
            q_length,
            self.head_dim,
        ]);
        let key_layer =
            key_layer
                .transpose(1, 2)
                .reshape([batch_size * self.num_kv, q_length, self.head_dim]);
        let value_layer = value_layer.transpose(1, 2).reshape([
            batch_size * self.num_kv,
            q_length,
            self.head_dim,
        ]);

        let (query_layer, key_layer) = if let Some(rotary) = self.maybe_rotary.borrow_mut() {
            let (query_layer, key_layer) = rotary.forward(&query_layer, &key_layer);
            (query_layer, key_layer)
        } else {
            (query_layer, key_layer)
        };

        let (key_layer, value_layer) = if let Some((past_key, past_value)) = layer_past {
            let key_layer = Tensor::cat(&[past_key, &key_layer], 1);
            let value_layer = Tensor::cat(&[past_value, &value_layer], 1);
            (key_layer, value_layer)
        } else {
            (key_layer, value_layer)
        };

        let kv_length = key_layer.size()[1];

        let present = if use_cache {
            Some((key_layer.shallow_clone(), value_layer.shallow_clone()))
        } else {
            None
        };

        if alibi.is_none() {
            let query_layer_ = query_layer.reshape([batch_size, self.num_heads, -1, self.head_dim]);
            let key_layer_ = key_layer.reshape([batch_size, self.num_kv, -1, self.head_dim]);
            let value_layer_ = value_layer.reshape([batch_size, self.num_kv, -1, self.head_dim]);
            let attn_output = tch::Tensor::scaled_dot_product_attention::<Tensor>(
                &query_layer_,
                &key_layer_,
                &value_layer_,
                None,
                0.0,
                true,
            );
            let x = attn_output.view([batch_size, self.num_heads, q_length, self.head_dim]);
            let x = x.permute([0, 2, 1, 3]);
            let attn_output = x.reshape([batch_size, q_length, self.num_heads * self.head_dim]);
            let attn_output = self.dense.forward(&attn_output);

            (attn_output, present)
        } else {
            let alibi = alibi.as_ref().unwrap();
            let attention_mask_float = (attention_mask * 1.0)
                .masked_fill(attention_mask, -1e9)
                .to_kind(Kind::BFloat16);
            let matmul_result = query_layer.matmul(&key_layer.transpose(1, 2));
            let attention_scores =
                matmul_result.view([batch_size, self.num_heads, q_length, kv_length]);
            let attention_scores = attention_scores.to_kind(Kind::Float);
            let attention_probs = (attention_scores
                + alibi.view([batch_size, self.num_heads, 1, -1]) * self.inv_norm_factor
                + attention_mask_float)
                .softmax(-1, hidden_states.kind());
            let mut attention_probs = self.attention_dropout.forward(&attention_probs);
            if !head_mask.is_none() {
                attention_probs *= head_mask.as_ref().unwrap();
            }
            let attention_probs_reshaped =
                attention_probs.view([batch_size * self.num_heads, q_length, kv_length]);
            let context_layer = attention_probs_reshaped.matmul(&value_layer);
            let context_layer = self.merge_heads(&context_layer);
            let output_tensor = self.dense.forward(&context_layer);
            let mut outputs = (output_tensor, present);
            if output_attentions {
                outputs.0 += attention_probs;
            }
            outputs
        }
    }
}

#[derive(Debug)]
pub struct MLP {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    // hidden_dropout: Dropout,
}

impl MLP {
    fn new(path: &Path, config: &RWConfig) -> Self {
        let hidden_size = config.hidden_size;
        let linear_config = LinearConfig {
            bias: config.bias,
            ..Default::default()
        };
        let dense_h_to_4h = linear(
            path / "dense_h_to_4h",
            hidden_size,
            4 * hidden_size,
            linear_config,
        );
        let dense_4h_to_h = linear(
            path / "dense_4h_to_h",
            4 * hidden_size,
            hidden_size,
            linear_config,
        );
        Self {
            dense_h_to_4h,
            dense_4h_to_h,
        }
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.dense_h_to_4h.forward(x);
        let x = x.gelu("none");
        self.dense_4h_to_h.forward(&x)
    }
}

#[derive(Debug)]
struct DecoderLayer {
    input_layernorm: LayerNorm,
    self_attention: Attention,
    config: RWConfig,
    training: bool,
    post_attention_layernorm: Option<LayerNorm>,
    mlp: MLP,
}

impl DecoderLayer {
    fn new(path: &Path, config: &RWConfig, training: bool) -> Self {
        let hidden_size = config.hidden_size;
        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_epsilon,
            ..Default::default()
        };
        let input_layernorm = layer_norm(
            path / "input_layernorm",
            vec![hidden_size],
            layer_norm_config,
        );
        // let num_heads = config.n_head;
        let self_attention = Attention::new(&(path / "self_attention"), config);

        let post_attention_layernorm = if !config.parallel_attn {
            Some(layer_norm(
                path / "post_attention_layernorm",
                vec![hidden_size],
                layer_norm_config,
            ))
        } else {
            None
        };

        let mlp = MLP::new(&(path / "mlp"), config);

        // let apply_residual_connection_post_layernorm =
        //     config.apply_residual_connection_post_layernorm;

        Self {
            input_layernorm,
            self_attention,
            config: config.clone(),
            training,
            post_attention_layernorm,
            mlp,
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        alibi: &Option<Tensor>,
        attention_mask: &Tensor,
        layer_past: &Option<(Tensor, Tensor)>,
        head_mask: &Option<Tensor>,
        use_cache: bool,
        output_attentions: bool,
    ) -> (Tensor, Option<(Tensor, Tensor)>) {
        let mut layernorm_output = self.input_layernorm.forward(hidden_states);
        let mut residual = hidden_states.shallow_clone();

        let attn_outputs = self.self_attention.forward(
            &layernorm_output,
            alibi,
            attention_mask,
            layer_past,
            head_mask,
            use_cache,
            output_attentions,
        );

        // attn output
        let attention_output = attn_outputs.0;

        if !self.config.parallel_attn {
            residual = dropout_add(
                &attention_output,
                &residual,
                self.config.attention_dropout,
                self.training,
            );

            layernorm_output = self
                .post_attention_layernorm
                .as_ref()
                .unwrap()
                .forward(&residual);
        }

        // key value output
        let outputs = attn_outputs.1;

        let mut mlp_output = self.mlp.forward(&layernorm_output);

        if self.config.parallel_attn {
            mlp_output += attention_output;
        }

        let output = dropout_add(
            &mlp_output,
            &residual,
            self.config.hidden_dropout,
            self.training,
        );

        if use_cache {
            (output, outputs)
        } else {
            (output, None::<(Tensor, Tensor)>)
        }
    }
}

#[derive(Debug)]
pub struct RWModel {
    config: RWConfig,
    h: Vec<DecoderLayer>,
}

impl RWModel {
    pub fn new(path: &Path, config: &RWConfig, training: bool) -> Self {
        let embed_dim = config.hidden_size;
        let num_heads = config.n_head;
        let alibi = config.alibi;

        let word_embeddings = embedding(
            path / "word_embeddings", // TODO: check if this is correct
            config.vocab_size,
            embed_dim,
            EmbeddingConfig::default(),
        );

        let mut h = Vec::<DecoderLayer>::with_capacity(config.get_num_hidden_layers() as usize);

        for i in 0..config.get_num_hidden_layers() {
            h.push(DecoderLayer::new(
                &(path / format!("decoder.layers.{}", i)), // TODO: check if this is correct
                config,
                training,
            ));
        }

        let ln_f = layer_norm(
            path / "ln_f",
            vec![embed_dim],
            LayerNormConfig {
                eps: config.layer_norm_epsilon,
                ..Default::default()
            },
        );
        Self {
            config: config.clone(),
            h,
        }
    }

    fn prepare_attn_mask(
        &self,
        attention_mask: &Tensor,
        input_shape: (i64, i64),
        past_key_values_length: i64,
    ) -> Tensor {
        let device = attention_mask.device();
        let (_, src_length) = input_shape;
        let combined_attention_mask = if src_length > 1 {
            Some(make_causal_mask(
                input_shape,
                device,
                past_key_values_length,
            ))
        } else {
            None
        };
        let expanded_attn_mask = expand_mask(attention_mask, &Some(src_length));

        combined_attention_mask
            .map(|combined_attention_mask| {
                expanded_attn_mask.bitwise_or_tensor(&combined_attention_mask)
            })
            .unwrap_or(expanded_attn_mask)
    }

    fn convert_head_mask_to_5d(&self, head_mask: &Tensor, num_hidden_layers: i64) -> Tensor {
        let head_mask = {
            if head_mask.dim() == 1 {
                let head_mask = head_mask
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .unsqueeze(-1);
                head_mask.expand([num_hidden_layers, -1, -1, -1, -1], false)
            } else if head_mask.dim() == 2 {
                head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            } else {
                head_mask.shallow_clone()
            }
        };

        assert!(
            head_mask.dim() == 5,
            "head_mask.dim() should be 5, but is {}",
            head_mask.dim()
        );

        head_mask.to_kind(Kind::Float)
    }

    fn get_head_mask(
        &self,
        head_mask: &Option<Tensor>,
        num_hidden_layers: i64,
        is_attention_chunked: bool,
    ) -> Tensor {
        if !head_mask.is_none() {
            let mut head_mask =
                self.convert_head_mask_to_5d(head_mask.as_ref().unwrap(), num_hidden_layers);

            if is_attention_chunked {
                head_mask = head_mask.unsqueeze(-1);
            }
            head_mask
        } else {
            Tensor::from_slice(&vec![0; num_hidden_layers as usize])
        }
    }

    fn forward(
        &self,
        input_ids: &Option<Tensor>,
        past_key_values: &Option<Vec<(Tensor, Tensor)>>,
        attention_mask: &Option<Tensor>,
        head_mask: &Option<Tensor>,
        inputs_embds: &Option<Tensor>,
        use_cache: Option<bool>,
        output_attentions: Option<bool>,
        output_hidden_states: Option<bool>,
        return_dict: Option<bool>,
    ) {
        let output_attentions = output_attentions.unwrap_or(false);
        let output_hidden_states = output_hidden_states.unwrap_or(false);
        let use_cache = use_cache.unwrap_or(self.config.use_cache);
        let return_dict = return_dict.unwrap_or(false);

        if input_ids.is_some() && inputs_embds.is_some() {
            panic!("Only one of input_ids or inputs_embeds may be set");
        }

        let (batch_size, seq_length) = if let Some(input_ids) = input_ids {
            input_ids.size2().unwrap()
        } else if let Some(inputs_embds) = inputs_embds {
            inputs_embds.size2().unwrap()
        } else {
            panic!("You have to specify either input_ids or inputs_embeds");
        };

        let past_key_values = past_key_values
            .as_ref()
            .unwrap_or(&Vec::with_capacity(self.h.len()));
    }
}

#[test]
fn test_i() {
    use tch::TensorIndexer;

    let unbounded: TensorIndexer = (..).into();

    println!("{:?}", unbounded);

    let bounded: TensorIndexer = (0..10).into();

    println!("{:?}", bounded);

    let half_bounded: TensorIndexer = (..10).into();

    println!("{:?}", half_bounded);

    let tensor_selector: TensorIndexer = (&Tensor::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9])).into();

    println!("{:?}", tensor_selector);

    let num_selector: TensorIndexer = 10.into();

    println!("{:?}", num_selector);

    let new_axis_selector: TensorIndexer = NewAxis.into();

    println!("{:?}", new_axis_selector);
}
