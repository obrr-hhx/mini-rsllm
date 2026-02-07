#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::Path;

use mini_rsllm::backend::{build_backend, DeviceKind};
use mini_rsllm::gguf::{GgufDType, GgufFile, TensorInfo};
use mini_rsllm::model::LlamaModel;
use mini_rsllm::tensor::TensorRef;
use mini_rsllm::tokenizer::Tokenizer;

const DEFAULT_MODEL_PATH: &str = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn make_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let x = xorshift64(&mut s) as f32 / u64::MAX as f32;
        out.push((x * 2.0 - 1.0) * scale);
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax(values: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

fn load_model_path() -> Option<String> {
    if let Ok(path) = std::env::var("MINIRSLLM_TEST_MODEL") {
        if Path::new(&path).exists() {
            return Some(path);
        }
    }
    if Path::new(DEFAULT_MODEL_PATH).exists() {
        return Some(DEFAULT_MODEL_PATH.to_string());
    }
    None
}

#[test]
fn operator_parity_cpu_vs_metal() {
    let cpu = build_backend(DeviceKind::Cpu, 0).unwrap();
    let metal = match build_backend(DeviceKind::Metal, 9999) {
        Ok(v) => v,
        Err(_) => return,
    };

    for i in 0..48u64 {
        let mut softmax_cpu = make_vec(i * 31 + 7, 257, 5.0);
        let mut softmax_metal = softmax_cpu.clone();
        cpu.softmax(&mut softmax_cpu);
        metal.softmax(&mut softmax_metal);
        assert!(
            max_abs_diff(&softmax_cpu, &softmax_metal) < 2e-5,
            "softmax mismatch at case {}",
            i
        );
    }

    for i in 0..32u64 {
        let x = make_vec(i * 19 + 3, 256, 2.0);
        let w = make_vec(i * 23 + 5, 256, 1.0);
        let cpu_out = cpu.rms_norm(&x, &w, 1e-5);
        let metal_out = metal.rms_norm(&x, &w, 1e-5);
        assert!(
            max_abs_diff(&cpu_out, &metal_out) < 2e-5,
            "rms_norm mismatch at case {}",
            i
        );
    }

    let mut rope_q_cpu = make_vec(101, 64, 1.0);
    let mut rope_k_cpu = make_vec(103, 32, 1.0);
    let mut rope_q_metal = rope_q_cpu.clone();
    let mut rope_k_metal = rope_k_cpu.clone();
    cpu.rope(&mut rope_q_cpu, &mut rope_k_cpu, 17, 16, 10000.0);
    metal.rope(&mut rope_q_metal, &mut rope_k_metal, 17, 16, 10000.0);
    assert!(
        max_abs_diff(&rope_q_cpu, &rope_q_metal) < 2e-5,
        "rope(q) mismatch"
    );
    assert!(
        max_abs_diff(&rope_k_cpu, &rope_k_metal) < 2e-5,
        "rope(k) mismatch"
    );
}

#[test]
fn matmul_f32_parity_cpu_vs_metal() {
    let cpu = build_backend(DeviceKind::Cpu, 0).unwrap();
    let metal = match build_backend(DeviceKind::Metal, 9999) {
        Ok(v) => v,
        Err(_) => return,
    };

    let rows = 64usize;
    let cols = 128usize;
    let mat = make_vec(2024, rows * cols, 1.0);
    let vec = make_vec(2025, cols, 0.5);

    let mut bytes = Vec::with_capacity(mat.len() * std::mem::size_of::<f32>());
    for &v in &mat {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    let info = TensorInfo {
        name: "test_mat".to_string(),
        shape: vec![cols, rows],
        dtype: GgufDType::F32,
        offset: 0,
    };
    let tensor = TensorRef { data: &bytes, info };

    let cpu_out = cpu.matmul_vec(&tensor, &vec);
    let metal_out = metal.matmul_vec(&tensor, &vec);
    assert!(
        max_abs_diff(&cpu_out, &metal_out) < 2e-4,
        "matmul f32 mismatch"
    );
}

#[test]
fn tinyllama_greedy_decode_matches_cpu() {
    let model_path = match load_model_path() {
        Some(v) => v,
        None => return,
    };

    let gguf = GgufFile::open(&model_path).expect("failed to open GGUF");
    let tokenizer = Tokenizer::from_gguf(&gguf);
    let cpu_backend = build_backend(DeviceKind::Cpu, 0).unwrap();
    let metal_backend = match build_backend(DeviceKind::Metal, 9999) {
        Ok(v) => v,
        Err(_) => return,
    };

    let mut cpu_model = LlamaModel::from_gguf_with_backend(&gguf, cpu_backend);
    let mut metal_model = LlamaModel::from_gguf_with_backend(&gguf, metal_backend);

    let prompt_tokens = tokenizer.encode("Hello from Apple Silicon metal parity", true);
    assert!(
        !prompt_tokens.is_empty(),
        "prompt tokens should not be empty"
    );

    let mut pos = 0usize;
    let mut cpu_next = 0u32;
    let mut metal_next = 0u32;
    let mut matched = 0usize;

    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let cpu_logits = cpu_model.forward(tok, pos).expect("cpu forward failed");
        let metal_logits = metal_model.forward(tok, pos).expect("metal forward failed");
        if argmax(&cpu_logits) == argmax(&metal_logits) {
            matched += 1;
        }
        if i + 1 == prompt_tokens.len() {
            cpu_next = argmax(&cpu_logits);
            metal_next = argmax(&metal_logits);
        }
        pos += 1;
    }

    let prompt_ratio = matched as f32 / prompt_tokens.len() as f32;
    assert!(
        prompt_ratio >= 0.99,
        "prompt top-1 parity too low: {:.4}",
        prompt_ratio
    );

    let decode_steps = 6usize.min(cpu_model.config.max_seq_len.saturating_sub(pos));
    assert!(decode_steps > 0, "decode steps should be > 0");

    let mut cpu_seq = Vec::with_capacity(decode_steps);
    let mut metal_seq = Vec::with_capacity(decode_steps);
    for _ in 0..decode_steps {
        cpu_seq.push(cpu_next);
        metal_seq.push(metal_next);
        let cpu_logits = cpu_model
            .forward(cpu_next, pos)
            .expect("cpu decode forward failed");
        let metal_logits = metal_model
            .forward(metal_next, pos)
            .expect("metal decode forward failed");
        cpu_next = argmax(&cpu_logits);
        metal_next = argmax(&metal_logits);
        pos += 1;
    }

    assert_eq!(cpu_seq, metal_seq, "greedy decode token sequence mismatch");
}
