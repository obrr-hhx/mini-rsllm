// mini-rsllm: A minimal Rust LLM inference engine
// CLI entry point with arg parsing and generation loop.

mod gguf;
mod model;
mod sampler;
mod tensor;
mod tokenizer;

use std::io::{self, Write};
use std::time::Instant;

use model::LlamaModel;
use sampler::{Sampler, SamplerConfig};
use tokenizer::Tokenizer;

struct Args {
    model_path: String,
    prompt: String,
    n_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    seed: u64,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: mini-rsllm <model.gguf> [options]");
        eprintln!("Options:");
        eprintln!("  -p, --prompt <text>     Input prompt (default: \"\")");
        eprintln!("  -n, --n-tokens <N>      Max tokens to generate (default: 256)");
        eprintln!("  -t, --temperature <F>   Sampling temperature (default: 0.8)");
        eprintln!("  --top-p <F>             Top-p sampling (nucleus). Keep smallest set with cumulative prob ≥ p (default: 0.9)");
        eprintln!("  --top-k <N>             Top-k sampling. Keep top k tokens by logit (0 = disable) (default: 40)");
        eprintln!("  --seed <N>              Random seed (default: 42)");
        std::process::exit(1);
    }

    let mut result = Args {
        model_path: args[1].clone(),
        prompt: String::new(),
        n_tokens: 256,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        seed: 42,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--prompt" => {
                i += 1;
                result.prompt = args[i].clone();
            }
            "-n" | "--n-tokens" => {
                i += 1;
                result.n_tokens = args[i].parse().expect("invalid n-tokens");
            }
            "-t" | "--temperature" => {
                i += 1;
                result.temperature = args[i].parse().expect("invalid temperature");
            }
            "--top-p" => {
                i += 1;
                result.top_p = args[i].parse().expect("invalid top-p");
            }
            "--top-k" => {
                i += 1;
                result.top_k = args[i].parse().expect("invalid top-k");
            }
            "--seed" => {
                i += 1;
                result.seed = args[i].parse().expect("invalid seed");
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    result
}

fn main() {
    let args = parse_args();

    // 1. Load GGUF file
    let load_start = Instant::now();
    let gguf = gguf::GgufFile::open(&args.model_path).expect("failed to open GGUF file");
    let _load_time = load_start.elapsed().as_secs_f64();

    // 2. Build tokenizer
    let tokenizer = Tokenizer::from_gguf(&gguf);
    let _vocab_size = tokenizer.vocab_size();

    // 3. Build model
    let mut model = LlamaModel::from_gguf(&gguf);

    // 4. Encode prompt
    let prompt_tokens = tokenizer.encode(&args.prompt, true);
    let _prompt_len = args.prompt.len();

    // 5. Set up sampler
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    });

    let total_start = Instant::now();
    let mut pos = 0;
    let mut token = prompt_tokens[0];
    let mut n_generated = 0;
    let gen_start;

    // 6. Prefill: process prompt tokens
    let prefill_start = Instant::now();
    for i in 0..prompt_tokens.len() {
        let mut logits = model.forward(prompt_tokens[i], pos);
        pos += 1;

        if i == prompt_tokens.len() - 1 {
            // Sample next token from the last prompt position
            token = sampler.sample(&mut logits);
        }
    }
    let _prefill_time = prefill_start.elapsed().as_secs_f64();

    // 7. Generate: sample and print tokens
    gen_start = Instant::now();
    for step in 0..args.n_tokens {
        // Print the token
        let text = tokenizer.decode_token(token);
        print!("{}", text);
        io::stdout().flush().unwrap();

        // Check for EOS
        if token == tokenizer.eos_id {
            break;
        }

        // Forward pass for the next token
        let mut logits = model.forward(token, pos);
        pos += 1;
        n_generated += 1;

        // Sample next token
        token = sampler.sample(&mut logits);
    }

    println!();
    let _gen_time = gen_start.elapsed().as_secs_f64();
    let _total_time = total_start.elapsed().as_secs_f64();
    let _ = n_generated;
}
