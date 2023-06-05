mod config;
mod loader;
mod model;

use anyhow::Result;

fn main() -> Result<()> {
    // let tensor = loader::load_multiz().unwrap();

    let cwd = std::env::current_dir()?;
    let path = cwd.join("pytorch_model.bin");
    let config_path = cwd.join("1B.config.json");

    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);

    // println!("{:?}", path.as_path());

    // vs.load(path.as_path())?;

    // let binding = vs.variables();
    // // let var = binding.keys().collect::<Vec<_>>();

    // println!("{}", binding.len());

    let file = std::fs::File::open(config_path)?;

    let config: config::RWConfig = serde_json::from_reader(file)?;

    model::RWForCausalLM::new(&vs.root(), &config, false);

    let binding = vs.variables();
    let keys = binding.keys().collect::<Vec<_>>();

    println!("{:?}", keys);

    Ok(())
}

#[test]
fn test() {
    main();
}
