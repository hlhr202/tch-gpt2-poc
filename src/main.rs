mod attention;
mod config;
mod transformers;

use std::collections::HashMap;

use config::Config;

use transformers::GPT2LMHeadModel;

use anyhow::Result;
use tch::Tensor;

/* fn get_device() -> tch::Device {
    if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
} */

fn main() -> Result<()> {
    let cwd = std::env::current_dir().unwrap();
    let path = cwd.join("model.safetensors");

    let mut tensors = Tensor::read_safetensors(path)?;

    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    let tensor_map = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor))
        .collect::<HashMap<String, &Tensor>>();

    let model = GPT2LMHeadModel::new(
        &tensor_map,
        &Config {
            n_layer: 12,
            n_embd: 768,
            n_head: 12,
            layer_norm_epsilon: Some(1e-5),
        },
    );

    println!("{:#?}", model);

    Ok(())
}
