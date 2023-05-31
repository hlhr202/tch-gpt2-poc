use std::collections::HashMap;

use tch::{nn::Module, Tensor};

use crate::config::Config;

#[derive(Debug)]
pub struct Conv1D {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Module for Conv1D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.weight) + &self.bias
    }
}

#[derive(Debug)]
pub struct Attention {
    bias: Tensor,
    c_attn: Conv1D,
    c_proj: Conv1D,
    n_head: i64,
    n_state: i64,
    scale: bool,
}

impl Attention {
    pub fn from_path(
        path: &str,
        tensor_map: &HashMap<String, &Tensor>,
        config: &Config,
        scale: bool,
    ) -> Self {
        let bias = tensor_map
            .get(&format!("{}.bias", path))
            .unwrap_or_else(|| panic!("failed to get {}.bias", path));
        let c_attn_weight = tensor_map
            .get(&format!("{}.c_attn.weight", path))
            .unwrap_or_else(|| panic!("failed to get {}.c_attn.weight", path));
        let c_attn_bias = tensor_map
            .get(&format!("{}.c_attn.bias", path))
            .unwrap_or_else(|| panic!("failed to get {}.c_attn.bias", path));
        let c_proj_weight = tensor_map
            .get(&format!("{}.c_proj.weight", path))
            .unwrap_or_else(|| panic!("failed to get {}.c_proj.weight", path));
        let c_proj_bias = tensor_map
            .get(&format!("{}.c_proj.bias", path))
            .unwrap_or_else(|| panic!("failed to get {}.c_proj.bias", path));

        let c_attn = Conv1D {
            weight: c_attn_weight.shallow_clone(),
            bias: c_attn_bias.shallow_clone(),
        };

        let c_proj = Conv1D {
            weight: c_proj_weight.shallow_clone(),
            bias: c_proj_bias.shallow_clone(),
        };

        Self {
            bias: bias.shallow_clone(),
            c_attn,
            c_proj,
            n_head: config.n_head,
            n_state: config.n_embd,
            scale,
        }
    }

    fn split_heads(&self, x: &Tensor, k: bool) -> Tensor {
        let x = x.view((x.size()[0], -1, self.n_head, self.n_state / self.n_head));
        if k {
            x.permute([0, 2, 3, 1])
        } else {
            x.permute([0, 2, 1, 3])
        }
    }

    fn flatten(&self, x: &Tensor) -> Tensor {
        x.transpose(1, 2).contiguous().view((
            x.size()[0],
            -1,
            self.n_head * (self.n_state / self.n_head),
        ))
    }

    fn attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let mut w = query.matmul(key);
        if self.scale {
            w /= (*value
                .size()
                .last()
                .unwrap_or_else(|| panic!("failed to get last value in attention"))
                as f64)
                .sqrt();
        }

        let (nd, ns) = (w.size()[2], w.size()[3]);
        let b = self.bias.narrow(2, ns - nd, nd).narrow(3, 0, ns);
        let mut w: Tensor = w * &b + 1e4 * (&b - 1);
        if let Some(mask) = attention_mask {
            w += mask;
        }
        w = w.softmax(-1, w.kind());

        w.matmul(value)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        layer_past: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let x = x.apply(&self.c_attn).split(self.n_state, 2);
        let (query, key, value) = (
            self.split_heads(&x[0], false),
            self.split_heads(&x[1], true),
            self.split_heads(&x[2], false),
        );
        let (key, value) = match layer_past {
            Some(layer_past) => {
                let key = Tensor::cat(&[layer_past.get(0).transpose(-2, -1), key], -1);
                let value = Tensor::cat(&[layer_past.get(1), value], -2);
                (key, value)
            }
            None => (key, value),
        };

        let present = Tensor::stack(&[key.transpose(-2, -1), value.shallow_clone()], 0);
        let a = self.attention(&query, &key, &value, attention_mask);
        let a = self.flatten(&a).apply(&self.c_proj);

        (a, present)
    }
}
