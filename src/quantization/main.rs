use regex::Regex;
use tch::{Kind, Tensor};

fn quantize_tensors() -> Vec<Regex> {
    [
        "wte",
        "lm_head",
        "h.*.attn.c_attn.weight",
        "h.*.attn.c_proj.weight",
        "h.*.mlp.c_fc.weight",
        "h.*.mlp.c_proj.weight",
    ]
    .into_iter()
    .map(|s| Regex::new(s).unwrap())
    .collect()
}

fn main() {
    let cwd = std::env::current_dir().unwrap();
    let model_path = cwd.join("model.safetensors");
    let save_path = cwd.join("model_qint8.ot");

    let mut tensors = Tensor::read_safetensors(model_path).unwrap();

    let tensors = tensors
        .iter_mut()
        .map(|(name, tensor)| {
            let to_quantize = quantize_tensors();
            let quantized = to_quantize.iter().any(|q| q.is_match(name)) && tensor.dim() == 2;
            if quantized {
                println!("type: {:?}, name: {}", tensor.kind(), name);
                let quantized_tensor = tensor.quantize_per_tensor_dynamic(Kind::QInt8, false);
                (name, quantized_tensor)
            } else {
                (name, tensor.shallow_clone())
            }
        })
        .collect::<Vec<_>>();

    Tensor::save_multi(&tensors, save_path).unwrap();
}
