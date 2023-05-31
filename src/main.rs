mod attention;
mod config;
mod transformers;

use std::{collections::HashMap, io::Write};

use config::Config;

use transformers::GPT2LMHeadModel;

use tch::Tensor;
use tokenizers::tokenizer::{Result, Tokenizer};

/* fn get_device() -> tch::Device {
    if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
} */

fn top_k_logits(logits: &Tensor, k: i64) -> Tensor {
    if k <= 0 {
        return logits.shallow_clone();
    }
    let (values, _) = logits.topk(k, -1, true, false);
    let min_values = values.get(-1);
    logits.where_self(
        &logits.lt_tensor(&min_values),
        &(Tensor::ones_like(logits) * f64::NEG_INFINITY),
    )
}

// TODO: not sure if this part is working well
fn sample_logits(logits: &Tensor) -> i64 {
    let logits = top_k_logits(logits, 1);
    let probs = logits.softmax(-1, logits.kind());
    let probs = probs.multinomial(1, true);
    i64::try_from(probs).unwrap()
}

fn main() -> Result<()> {
    let cwd = std::env::current_dir()?;
    let path = cwd.join("model.safetensors");

    let mut tensors = Tensor::read_safetensors(path)?;

    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    let tensor_map = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor))
        .collect::<HashMap<String, &Tensor>>();

    let tokenizer = Tokenizer::from_pretrained("gpt2", None)?;

    let prompt = "My name is Merve and my favorite";

    print!("{}", prompt);

    std::io::stdout().flush().unwrap();

    let input = tokenizer.encode(prompt, false)?;

    let mut input_ids = input
        .get_ids()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();

    let model = GPT2LMHeadModel::new(
        &tensor_map,
        &Config {
            n_layer: 12,
            n_embd: 768,
            n_head: 12,
            layer_norm_epsilon: Some(1e-5),
        },
    );

    let num_predict = 256;

    let mut current = 0;

    while current < num_predict {
        current += 1;

        let batch_input = Tensor::from_slice(&input_ids).reshape([1, -1]);

        let (logits, _past) = model.forward(&batch_input, None, None, None, None);

        let logits = logits.select(1, -1).squeeze();
        let token = sample_logits(&logits);

        input_ids.push(token);

        let str = tokenizer.decode([token as u32].to_vec(), false).unwrap();

        print!("{}", str);

        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
