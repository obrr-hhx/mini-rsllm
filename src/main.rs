// mini-rsllm: A minimal Rust LLM inference engine
// CLI entry point with arg parsing and generation loop.

mod backend;
mod gguf;
#[cfg(feature = "metal")]
mod metal;
mod model;
mod sampler;
mod tensor;
mod tokenizer;

use std::io::{self, Write};
use std::time::Instant;

use backend::{build_backend, DeviceKind};
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
    device: DeviceKind,
    gpu_layers: usize,
}

fn print_usage() {
    eprintln!("Usage: mini-rsllm <model.gguf> [options]");
    eprintln!("Options:");
    eprintln!("  -p, --prompt <text>     Input prompt (default: \"\")");
    eprintln!("  -n, --n-tokens <N>      Max tokens to generate (default: 256)");
    eprintln!("  -t, --temperature <F>   Sampling temperature (default: 0.8)");
    eprintln!("  --top-p <F>             Top-p sampling (0, 1] (default: 0.9)");
    eprintln!("  --top-k <N>             Top-k sampling (0 = disable) (default: 40)");
    eprintln!("  --seed <N>              Random seed (default: 42)");
    eprintln!("  --device <cpu|metal>    Runtime backend device (default: cpu)");
    eprintln!(
        "  --gpu-layers <N>        Number of transformer layers assigned to GPU (default: 0)"
    );
}

fn require_value(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    if *i >= args.len() {
        eprintln!("Missing value for {}", flag);
        std::process::exit(1);
    }
    args[*i].clone()
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
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
        device: DeviceKind::Cpu,
        gpu_layers: 0,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--prompt" => {
                let flag = args[i].as_str();
                result.prompt = require_value(&args, &mut i, flag);
            }
            "-n" | "--n-tokens" => {
                let flag = args[i].as_str();
                let raw = require_value(&args, &mut i, flag);
                result.n_tokens = raw.parse().unwrap_or_else(|_| {
                    eprintln!("invalid n-tokens: {}", raw);
                    std::process::exit(1);
                });
            }
            "-t" | "--temperature" => {
                let flag = args[i].as_str();
                let raw = require_value(&args, &mut i, flag);
                result.temperature = raw.parse().unwrap_or_else(|_| {
                    eprintln!("invalid temperature: {}", raw);
                    std::process::exit(1);
                });
            }
            "--top-p" => {
                let flag = args[i].as_str();
                let raw = require_value(&args, &mut i, flag);
                result.top_p = raw.parse().unwrap_or_else(|_| {
                    eprintln!("invalid top-p: {}", raw);
                    std::process::exit(1);
                });
            }
            "--top-k" => {
                let flag = args[i].as_str();
                let raw = require_value(&args, &mut i, flag);
                result.top_k = raw.parse().unwrap_or_else(|_| {
                    eprintln!("invalid top-k: {}", raw);
                    std::process::exit(1);
                });
            }
            "--seed" => {
                let flag = args[i].as_str();
                let raw = require_value(&args, &mut i, flag);
                result.seed = raw.parse().unwrap_or_else(|_| {
                    eprintln!("invalid seed: {}", raw);
                    std::process::exit(1);
                });
            }
            "--device" => {
                let flag = args[i].as_str();
                let raw = require_value(&args, &mut i, flag);
                result.device = raw.parse().unwrap_or_else(|msg: String| {
                    eprintln!("{}", msg);
                    std::process::exit(1);
                });
            }
            "--gpu-layers" => {
                let flag = args[i].as_str();
                let raw = require_value(&args, &mut i, flag);
                result.gpu_layers = raw.parse().unwrap_or_else(|_| {
                    eprintln!("invalid gpu-layers: {}", raw);
                    std::process::exit(1);
                });
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if result.temperature < 0.0 {
        eprintln!("temperature must be >= 0");
        std::process::exit(1);
    }
    if !(0.0 < result.top_p && result.top_p <= 1.0) {
        eprintln!("top-p must be in range (0, 1]");
        std::process::exit(1);
    }
    if result.device == DeviceKind::Cpu && result.gpu_layers > 0 {
        eprintln!("--gpu-layers is only valid when --device metal");
        std::process::exit(1);
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
    let backend = build_backend(args.device, args.gpu_layers).unwrap_or_else(|e| {
        eprintln!(
            "failed to initialize backend '{}': {}",
            args.device.as_str(),
            e
        );
        std::process::exit(1);
    });
    let mut model = LlamaModel::from_gguf_with_backend(&gguf, backend);

    // 4. Encode prompt
    let prompt_tokens = tokenizer.encode(&args.prompt, true);
    let _prompt_len = args.prompt.len();
    if prompt_tokens.is_empty() {
        eprintln!("prompt encoding produced no tokens");
        std::process::exit(1);
    }

    if prompt_tokens.len() > model.config.max_seq_len {
        eprintln!(
            "prompt is too long: {} tokens, context length is {}",
            prompt_tokens.len(),
            model.config.max_seq_len
        );
        std::process::exit(1);
    }

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
    let max_new_tokens = args
        .n_tokens
        .min(model.config.max_seq_len.saturating_sub(prompt_tokens.len()));
    if max_new_tokens < args.n_tokens {
        eprintln!(
            "warning: clipping n-tokens from {} to {} due to context length {}",
            args.n_tokens, max_new_tokens, model.config.max_seq_len
        );
    }

    eprintln!(
        "backend: {} (requested: {}, gpu-layers: {})",
        model.backend_name(),
        args.device.as_str(),
        args.gpu_layers
    );

    // 6. Prefill: process prompt tokens
    let prefill_start = Instant::now();
    for i in 0..prompt_tokens.len() {
        let mut logits = model.forward(prompt_tokens[i], pos).unwrap_or_else(|e| {
            eprintln!("forward failed at prompt step {}: {}", i, e);
            std::process::exit(1);
        });
        pos += 1;

        if i == prompt_tokens.len() - 1 {
            // Sample next token from the last prompt position
            token = sampler.sample(&mut logits);
        }
    }
    let _prefill_time = prefill_start.elapsed().as_secs_f64();

    // 7. Generate: sample and print tokens
    gen_start = Instant::now();
    for _step in 0..max_new_tokens {
        // Print the token
        let text = tokenizer.decode_token(token);
        print!("{}", text);
        io::stdout().flush().unwrap();

        // Check for EOS
        if token == tokenizer.eos_id {
            break;
        }

        // Forward pass for the next token
        let mut logits = model.forward(token, pos).unwrap_or_else(|e| {
            eprintln!("forward failed at generation position {}: {}", pos, e);
            std::process::exit(1);
        });
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
