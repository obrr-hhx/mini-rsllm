# mini-rsllm vs llama.cpp Comparison Report

Date: 2026-02-07  
Workspace: `/Users/huanghx/code/mini-rsllm`  
Model: `models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf`  
Prompt: `"Hello"`  
Generated tokens: `32`  

## 1) Test Setup

- `mini-rsllm` binary: `./target/release/mini-rsllm`
- `llama.cpp` binary: `/opt/homebrew/bin/llama-completion`
- `llama.cpp` version: `7950 (449ec2ab0)`
- Device: Apple Silicon Metal (logs report `Apple M2`)

## 2) Commands

### mini-rsllm (CPU + Metal in one run)

```bash
N_TOKENS=32 GPU_LAYERS=9999 \
bench/bench_metal.sh models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf "Hello"
```

### llama.cpp (CPU)

```bash
/usr/bin/time -p /opt/homebrew/bin/llama-completion \
  -m models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -p "Hello" -n 32 --temp 0 --top-k 40 --top-p 0.9 -s 42 \
  --no-display-prompt --no-warmup --gpu-layers 0 --simple-io
```

### llama.cpp (Metal)

```bash
/usr/bin/time -p /opt/homebrew/bin/llama-completion \
  -m models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -p "Hello" -n 32 --temp 0 --top-k 40 --top-p 0.9 -s 42 \
  --no-display-prompt --no-warmup --gpu-layers all --simple-io
```

## 3) Results

### Raw metrics

| Engine | Mode | real_s | tokens/s |
|---|---|---:|---:|
| mini-rsllm | CPU | 20.01 | 1.599 |
| mini-rsllm | Metal | 5.63 | 5.684 |
| llama.cpp | CPU | 1.63 | 19.632 |
| llama.cpp | Metal | 1.09 | 29.358 |

### Derived ratios

- `mini-rsllm` Metal vs CPU: `3.555x`
- `llama.cpp` Metal vs CPU: `1.495x`
- `llama.cpp` vs `mini-rsllm` (CPU): `12.278x`
- `llama.cpp` vs `mini-rsllm` (Metal): `5.165x`

## 4) Observations

1. Both engines benefit from Metal on this model and token count.
2. `mini-rsllm` currently shows larger relative Metal uplift (`3.555x`) because its CPU baseline is much slower.
3. In absolute throughput, `llama.cpp` is still significantly faster in both CPU and Metal modes.

## 5) Notes / Fairness Caveats

1. This comparison uses one run per mode (no repetition/averaging), so results include run-to-run noise.
2. `mini-rsllm` and `llama.cpp` are different implementations with different optimization levels (threading, kernels, runtime internals), so absolute numbers are not strictly apples-to-apples.
3. Throughput here is computed as `n_tokens / real_s` with `n_tokens=32`, including model runtime overhead per invocation.

