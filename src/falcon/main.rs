mod config;
mod loader;
mod model;

use anyhow::Result;

fn main() -> Result<()> {
    let cwd = std::env::current_dir()?;
    let path = cwd.join("pytorch_model.bin");
    let config_path = cwd.join("1B.config.json");

    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);

    let file = std::fs::File::open(config_path)?;

    let config: config::RWConfig = serde_json::from_reader(file)?;

    let causal_lm = model::RWForCausalLM::new(&vs.root(), &config, false);

    println!("{:#?}", causal_lm);

    vs.load(path)?;

    Ok(())
}
