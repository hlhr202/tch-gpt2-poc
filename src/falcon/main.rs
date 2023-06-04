mod config;
mod loader;
mod model;

fn main() {
    let tensor = loader::load_multiz().unwrap();

    for (k, v) in tensor {
        println!("{}: {:?}", k, v.size());
    }
}

#[test]
fn test() {
    main();
}