mod config;
pub mod device;
mod model;

use std::io::Write;

use anyhow::Result;
use device::get_device;
use tch::{Kind, Tensor};
use tokenizers::Tokenizer;

fn top_k_logits(logits: &Tensor, k: i64) -> Tensor {
    if k <= 0 {
        return logits.shallow_clone();
    }
    let (values, _) = logits.topk(k, -1, true, false);
    let min_values = values.get(-1);
    logits.where_self(
        &logits.lt_tensor(&min_values),
        &(Tensor::ones_like(logits) * f64::NEG_INFINITY).to_device(get_device()),
    )
}

// TODO: not sure if this part is working well
fn sample_logits(logits: &Tensor) -> i64 {
    let logits = top_k_logits(logits, 1);
    let probs = logits.softmax(-1, logits.kind());
    let probs = probs.multinomial(1, true);
    i64::try_from(probs).unwrap()
}

fn main() -> Result<(), anyhow::Error> {
    let cwd = std::env::current_dir()?;
    let path = cwd.join("pytorch_model.bin");
    let config_path = cwd.join("1B.config.json");
    let tokenizer =
        Tokenizer::from_pretrained("tiiuae/falcon-rw-1b", None).map_err(|e| anyhow::anyhow!(e))?;

    let mut vs = tch::nn::VarStore::new(get_device());

    let file = std::fs::File::open(config_path)?;

    let config: config::RWConfig = serde_json::from_reader(file)?;

    let mut causal_lm = model::RWForCausalLM::new(&vs.root(), &config, false);

    vs.set_kind(Kind::Half);

    vs.load(path)?;

    let prompt = "Hello my name is";

    let n_ctx = 256;

    let input = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!(e))?;

    let mut input_ids = input
        .get_ids()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();

    let mut remaining = n_ctx - input_ids.len();

    input_ids.resize(n_ctx, 0);

    while remaining > 0 {
        let cursor = input_ids.len() - remaining;
        let input_slice = &input_ids[0..cursor];
        let batch_input = Tensor::from_slice(input_slice)
            .reshape([1, -1])
            .to_device(get_device());

        let output = causal_lm.forward(
            &Some(batch_input),
            &None,
            &None,
            &None,
            &None,
            Some(false),
            Some(false),
            Some(false),
        );

        // let logits = output.logits.select(1, -1).squeeze();
        // let logits = logits.softmax(-1, logits.kind());
        // let (_v, idx) = logits.topk(40, -1, true, false);
        // let token = i64::try_from(&idx.get(0)).unwrap();

        let logits = output.logits.select(1, -1).squeeze();
        let token = sample_logits(&logits);

        if token == 0 || token >= 50256 {
            break;
        }

        input_ids[cursor] = token;

        let str = tokenizer.decode([token as u32].to_vec(), false).unwrap();

        print!("{}", str);

        std::io::stdout().flush().unwrap();

        remaining -= 1;
    }

    Ok(())
}
