use anyhow::Result;
use tch::Tensor;

pub fn load_multiz() -> Result<Vec<(String, Tensor)>> {
    let cwd = std::env::current_dir()?;
    let path = cwd.join("pytorch_model.bin");

    let tensors = Tensor::loadz_multi(path)?;

    Ok(tensors)
}
