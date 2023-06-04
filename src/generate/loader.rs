use anyhow::Result;
use tch::Tensor;

pub fn load_safetensors() -> Result<Vec<(String, Tensor)>> {
    let cwd = std::env::current_dir()?;
    let path = cwd.join("model.safetensors");

    let tensors = Tensor::read_safetensors(path)?;

    Ok(tensors)
}

pub fn load_multi() -> Result<Vec<(String, Tensor)>> {
    let cwd = std::env::current_dir()?;
    let path = cwd.join("model_qint8.ot");

    let tensors = Tensor::load_multi(path)?;

    Ok(tensors)
}
