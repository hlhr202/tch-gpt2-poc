use tch::Device;

#[derive(Debug, Clone)]
pub struct RWConfig {
    // default: 250880
    pub vocab_size: i64,

    // default: 64
    pub hidden_size: i64,

    // default: 2
    pub n_layer: i64,

    // default: 8
    pub n_head: i64,

    // default: 1e-5
    pub layer_norm_epsilon: f64,

    // default: 0.02
    pub initializer_range: f64,

    // default: true
    pub use_cache: bool,

    // default: 1
    pub bos_token_id: i64,

    // default: 2
    pub eos_token_id: i64,

    // default: false
    pub apply_residual_connection_post_layernorm: bool,

    // default: 0.0
    pub hidden_dropout: f64,

    // default: 0.0
    pub attention_dropout: f64,

    // default: false
    pub multi_query: bool,

    // default: false
    pub alibi: bool,

    // default: false
    pub bias: bool,

    // default: false
    pub parallel_attn: bool,
}

impl RWConfig {
    pub fn get_device(&self) -> Device {
        Device::cuda_if_available()
    }
    pub fn is_rotary(&self) -> bool {
        !self.alibi
    }
    pub fn get_num_hidden_layers(&self) -> i64 {
        self.n_layer
    }
}
