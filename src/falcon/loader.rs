use anyhow::Result;
use tch::Tensor;

pub fn load_multiz() -> Result<Vec<(String, Tensor)>> {
    let cwd = std::env::current_dir()?;
    let path = cwd.join("pytorch_model.bin");

    let tensors = Tensor::loadz_multi_with_device(path, tch::Device::Mps)?;

    Ok(tensors)
}

pub fn main() -> Result<()>{
    let tensor = load_multiz()?;
    
    let keys = tensor.iter().map(|(k, _)| k).collect::<Vec<_>>();

    for key in keys {
        println!("{}", key);
    }

    Ok(())
}