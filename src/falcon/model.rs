#![allow(clippy::too_many_arguments)]
use std::borrow::{Borrow, BorrowMut};

use crate::config::RWConfig;
use tch::{
    nn::{layer_norm, Init, LayerNorm, LayerNormConfig, LinearConfig, Module, Path},
    Device, Kind, Tensor,
};

fn dropout_add(x: &Tensor, residual: &Tensor, p: f64, train: bool) -> Tensor {
    let out = x.dropout(p, train);
    residual + out
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
        let num_heads = config.n_head;
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

        let apply_residual_connection_post_layernorm =
            config.apply_residual_connection_post_layernorm;

        let hidden_dropout = config.hidden_dropout;

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
    ) {
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

        let mut outputs = attn_outputs.1;

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

        // TODO: impl output
    }
}

#[test]
fn test() {
    let qkv = Tensor::randn([2, 3, 4], (Kind::Float, Device::Cpu));

    qkv.print();

    println!("-------------------");

    let q = qkv.index_select(-2, &Tensor::from_slice(&[0]));

    q.print();
}
