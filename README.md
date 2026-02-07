# mini-rsllm

A minimal LLM inference engine written from scratch in Rust. Loads real GGUF model files and generates text using the LLaMA architecture. Built for learning — clarity over performance, but fully functional with real models.

## Features

- **GGUF v2/v3 parser** with memory-mapped file I/O (no full model copy into RAM)
- **LLaMA architecture**: RMS normalization, RoPE, grouped-query attention (GQA), SwiGLU FFN
- **Quantization support**: F32, F16, Q4_0, Q8_0, Q6_K — dequantized on-the-fly during matrix-vector multiply
- **SentencePiece-style BPE tokenizer** loaded from GGUF metadata, with byte fallback
- **Sampling**: temperature, top-k, top-p (nucleus), with reproducible xorshift64 PRNG
- **Streaming output**: tokens printed as they are generated
- **Zero dependencies beyond 3 crates**: `memmap2`, `byteorder`, `half`

## Build

```
cargo build --release
```

## Usage

```
mini-rsllm <model.gguf> [options]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-p, --prompt <text>` | Input prompt | `""` |
| `-n, --n-tokens <N>` | Max tokens to generate | `256` |
| `-t, --temperature <F>` | Sampling temperature (0 = greedy) | `0.8` |
| `--top-p <F>` | Top-p / nucleus sampling | `0.9` |
| `--top-k <N>` | Top-k sampling (0 = disable) | `40` |
| `--seed <N>` | Random seed | `42` |
| `--device <cpu\|metal>` | 运行后端类型 | `cpu` |
| `--gpu-layers <N>` | 分配到 GPU 的 Transformer 层数（仅 Metal） | `0` |

### Examples

```bash
# Greedy decoding (deterministic)
./target/release/mini-rsllm model.gguf -p "The capital of France is" -n 30 -t 0.0

# Creative generation with sampling
./target/release/mini-rsllm model.gguf -p "Once upon a time" -n 100 -t 0.8

# Code completion
./target/release/mini-rsllm model.gguf -p "def fibonacci(n):" -n 50 -t 0.0
```

### Metal (Apple Silicon)

```bash
# 构建 Metal 版本
cargo build --release --features metal

# 使用 Metal，默认把最后 N 层放到 GPU（N=0 表示全 CPU）
./target/release/mini-rsllm model.gguf \
  --device metal \
  --gpu-layers 9999 \
  -p "Hello" -n 64 -t 0.0
```

CPU/Metal 对比基准：

```bash
tasks/bench_cpu.sh model.gguf "Hello"
tasks/bench_metal.sh model.gguf "Hello"
```

Metal 回归与稳定性测试：

```bash
# 关键算子 + 端到端 parity（CPU vs Metal）
cargo test --features metal --test metal_parity

# 长时间稳定性（输出一致性 + RSS 增长阈值）
scripts/soak_metal.sh model.gguf "Hello"
```

可选环境变量：

- `MINIRSLLM_TEST_MODEL`：指定 `tests/metal_parity.rs` 使用的模型路径
- `ITERATIONS`：soak 循环次数（默认 `20`）
- `MAX_RSS_GROWTH_KB`：允许的 RSS 增长阈值（默认 `262144`）

## Project Structure

```
src/
├── lib.rs         Library entry (for tests and reuse in bin)
├── main.rs        CLI entry point, arg parsing, generation loop
├── gguf.rs        GGUF v2/v3 binary format parser
├── tensor.rs      Tensor storage, dequantization, math operations
├── model.rs       LLaMA model: config, weights, KV cache, forward pass
├── tokenizer.rs   BPE tokenizer: encode text → tokens, decode tokens → text
├── sampler.rs     Temperature, top-k, top-p sampling
└── metal/         Metal backend, kernels, and bridge

tests/
└── metal_parity.rs  CPU vs Metal numerical parity and greedy decode parity

scripts/
└── soak_metal.sh  Metal soak test (stability + memory growth guard)
```

## How It Works

### GGUF Loading

The GGUF parser reads the binary header, metadata key-value pairs, and tensor descriptors. Tensor data stays memory-mapped — weights are never fully loaded into RAM. Each tensor is accessed as a byte slice into the mmap when needed.

### Forward Pass

For each token, the forward pass runs through all transformer layers:

1. **Embedding lookup** — index into the token embedding table
2. **For each layer:**
   - RMS normalize the hidden state
   - Project to Q, K, V via quantized matrix-vector multiply
   - Apply rotary position embeddings (RoPE) to Q and K
   - Store K, V into the KV cache
   - Multi-head attention with GQA (grouped-query attention)
   - Output projection + residual connection
   - RMS normalize, then SwiGLU FFN (gate/up/down projections) + residual
3. **Final RMS norm** and output projection to logits

All weight matrices are dequantized on-the-fly during dot products — one row at a time — so memory usage stays proportional to the model file size, not the full f32 parameter count.

### Quantization Formats

| Format | Block Size | Bytes/Block | Description |
|--------|-----------|-------------|-------------|
| F32 | 1 | 4 | Full precision |
| F16 | 1 | 2 | Half precision |
| Q8_0 | 32 | 34 | 8-bit quantization with f16 scale |
| Q4_0 | 32 | 18 | 4-bit quantization with f16 scale |
| Q6_K | 256 | 210 | 6-bit k-quant with per-block scales |

## Compatible Models

Any GGUF model using the LLaMA architecture with supported quantization types. Tested with:

- [TinyLlama-1.1B-Chat-v1.0 Q4_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)

Download a model and place it in the `models/` directory:

```bash
mkdir -p models
# Example: download TinyLlama Q4_0 (~600MB)
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_0.gguf --local-dir models/
```

> **Note**: Chat-finetuned models (like TinyLlama-Chat) work best with their expected chat template format. For raw text completion, base (non-chat) models will give more predictable results.

## CI

- `cpu-checks`（Linux）：`cargo fmt --check` + `cargo test --all-targets`
- `metal-smoke`（macOS）：`cargo build --release --features metal` + `cargo test --features metal --lib --tests`

## License

MIT
