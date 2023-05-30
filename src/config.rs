pub struct Config {
    pub n_layer: i64,
    pub n_embd: i64,
    pub n_head: i64,
    pub layer_norm_epsilon: Option<f64>, // 1e-5
}
