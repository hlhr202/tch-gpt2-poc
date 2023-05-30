mod attention;
mod config;
mod transformers;

use config::Config;
use std::{collections::HashMap, ops::Deref};

use anyhow::Result;
use tch::{
    nn::{self, Linear, Module},
    Kind, Tensor,
};

fn get_device() -> tch::Device {
    if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
}



fn main() -> Result<()> {
    let cwd = std::env::current_dir().unwrap();
    let path = cwd.join("model.safetensors");

    let mut tensors = Tensor::read_safetensors(path)?;

    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, tensor) in &tensors {
        println!("{}: {:?}", name, tensor.size());
    }

    let tensor_map = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor))
        .collect::<std::collections::HashMap<String, &Tensor>>();

    Ok(())
}
