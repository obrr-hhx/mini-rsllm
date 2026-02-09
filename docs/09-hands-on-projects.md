# 第 9 章：实践与扩展

> **学习目标**：通过动手项目巩固所学知识，探索扩展方向，培养独立修改推理引擎的能力。
>
> **预计时间**：10-15 小时（根据选择的项目而定）

---

## 9.1 实验环境准备

在开始实践之前，确保你有一个可运行的环境：

```bash
# 编译（Release 模式，否则太慢）
cargo build --release

# 验证可以运行
./target/release/mini-rsllm models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -p "Hello" -n 10 -t 0.0

# 运行测试
cargo test --all-targets
```

### 调试技巧

```rust
// 在关键位置添加 eprintln! 输出中间结果
eprintln!("[DEBUG] token={}, pos={}, logits[0..5]={:?}", token, pos, &logits[..5]);

// 使用 RUST_BACKTRACE 查看错误堆栈
// RUST_BACKTRACE=1 cargo run --release -- model.gguf -p "Hello"
```

---

## 9.2 项目一：采样策略实验（⭐ 入门）

**目标**：理解不同采样参数对生成质量的影响。

### 实验 1：温度对比

用相同的 prompt，不同温度生成文本：

```bash
# 贪婪解码（完全确定性）
./target/release/mini-rsllm model.gguf -p "The meaning of life is" -n 50 -t 0.0

# 低温度（保守）
./target/release/mini-rsllm model.gguf -p "The meaning of life is" -n 50 -t 0.3

# 中温度（平衡）
./target/release/mini-rsllm model.gguf -p "The meaning of life is" -n 50 -t 0.8

# 高温度（创造性）
./target/release/mini-rsllm model.gguf -p "The meaning of life is" -n 50 -t 1.5
```

**观察**：
- 温度 0.0 每次输出完全相同
- 温度越高，输出越多样但可能越不连贯
- 温度 > 1.0 时可能出现乱码

### 实验 2：Top-k 和 Top-p 对比

```bash
# 只用 top-k
./target/release/mini-rsllm model.gguf -p "Once upon a time" -n 50 -t 0.8 --top-k 5 --top-p 1.0

# 只用 top-p
./target/release/mini-rsllm model.gguf -p "Once upon a time" -n 50 -t 0.8 --top-k 0 --top-p 0.5

# 两者结合
./target/release/mini-rsllm model.gguf -p "Once upon a time" -n 50 -t 0.8 --top-k 40 --top-p 0.9
```

### 挑战：实现重复惩罚

在 `src/sampler.rs` 中添加重复惩罚（repetition penalty）：

```rust
// 在 sample() 方法中，temperature 之前添加：
fn apply_repetition_penalty(logits: &mut [f32], recent_tokens: &[u32], penalty: f32) {
    for &tok in recent_tokens {
        if (tok as usize) < logits.len() {
            // 如果 logit > 0，除以 penalty；如果 < 0，乘以 penalty
            if logits[tok as usize] > 0.0 {
                logits[tok as usize] /= penalty;
            } else {
                logits[tok as usize] *= penalty;
            }
        }
    }
}
```

**提示**：
- 需要在 `Sampler` 中维护一个最近 token 的窗口
- penalty = 1.0 表示不惩罚，> 1.0 表示惩罚重复
- 典型值：1.1 ~ 1.3

---

## 9.3 项目二：添加新的量化格式（⭐⭐ 中级）

**目标**：理解量化原理，实现 Q5_0 格式支持。

### Q5_0 格式规范

```
块大小：32 个元素
存储格式：
  [f16 scale] [4 bytes high-bits] [16 bytes low-nibbles]
  = 2 + 4 + 16 = 22 字节/块

每个元素 = 5 bit：
  - 4 bit 低位（与 Q4_0 相同，两个元素共享一个字节）
  - 1 bit 高位（32 个元素的高位打包在 4 字节中）

反量化：
  value = scale * (combined_5bit - 16)
  其中 combined_5bit = low_4bit | (high_1bit << 4)
```

### 实现步骤

**第 1 步**：在 `src/gguf.rs` 中添加类型

```rust
// GgufDType 枚举中添加
Q5_0 = 6,

// block_size() 中添加
GgufDType::Q5_0 => (22, 32),  // 22 字节, 32 元素
```

**第 2 步**：在 `src/tensor.rs` 中实现反量化

```rust
fn dequantize_q5_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 22;  // 2 (scale) + 4 (high bits) + 16 (low nibbles)
    let n_blocks = n / 32;
    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let block = &data[b * block_size..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let high_bits = &block[2..6];   // 4 bytes = 32 bits
        let low_nibs = &block[6..22];   // 16 bytes = 32 nibbles
```

---

## 9.4 项目三：性能分析与优化（⭐⭐ 中级）

**目标**：学会分析推理性能瓶颈，实施简单优化。

### 实验 1：基准测试

使用项目自带的基准脚本：

```bash
# CPU 基准
bench/bench_cpu.sh models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf "Hello"

# Metal 基准（需要 macOS + Apple Silicon）
cargo build --release --features metal
bench/bench_metal.sh models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf "Hello"
```

### 实验 2：逐层计时

在 `src/model.rs` 的 `forward()` 中添加计时：

```rust
use std::time::Instant;

// 在每个主要操作前后添加计时
let t0 = Instant::now();
let x_norm = backend.rms_norm(&x, &weights.rms_att_weight[layer], eps);
eprintln!("  layer {} rms_norm: {:.3}ms", layer, t0.elapsed().as_secs_f64() * 1000.0);

let t1 = Instant::now();
let q = backend.matmul_vec(&weights.wq[layer], &x_norm);
eprintln!("  layer {} wq matmul: {:.3}ms", layer, t1.elapsed().as_secs_f64() * 1000.0);
```

**你会发现**：
- `matmul_vec` 占据 90%+ 的时间
- `rms_norm`、`softmax` 等小算子几乎不耗时
- 注意力计算随序列长度线性增长

### 挑战：实现 SIMD 加速的点积

```rust
// 在 tensor.rs 中，替换朴素的点积实现
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    // 提示：使用 4 路展开减少循环开销
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
        sum2 += a[base + 2] * b[base + 2];
        sum3 += a[base + 3] * b[base + 3];
    }

    // 处理剩余元素
    let mut tail_sum = 0.0f32;
    for i in (chunks * 4)..a.len() {
        tail_sum += a[i] * b[i];
    }

    (sum0 + sum1) + (sum2 + sum3) + tail_sum
}
```

---

## 9.5 项目四：实现批量预填充（⭐⭐⭐ 高级）

**目标**：将逐 token 预填充改为批量处理，理解批量推理的优势。

### 当前实现的问题

```rust
// 当前：逐 token 预填充
for i in 0..prompt_tokens.len() {
    logits = model.forward(prompt_tokens[i], pos);
    pos += 1;
}
// 问题：每次 forward 只处理一个 token，无法利用矩阵并行性
```

### 改进方向

```rust
// 目标：批量预填充
fn forward_batch(&mut self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
    let seq_len = tokens.len();

    // 1. 嵌入：查找所有 token 的嵌入向量
    // 形状从 [dim] 变为 [seq_len, dim]

    // 2. 注意力：Q, K, V 都变成矩阵
    // matmul_vec 变为 matmul_mat

    // 3. KV 缓存：一次性写入多个位置

    // 4. 只返回最后一个位置的 logits
}
```

**实现提示**：
- 需要在 `tensor.rs` 中添加矩阵-矩阵乘法（`matmul_mat`）
- 需要修改 `Backend` trait 添加批量接口
- 注意力掩码需要变成因果掩码（causal mask）
- 这是一个较大的改动，建议在新分支上进行

---

## 9.6 项目五：CPU vs Metal 数值一致性验证（⭐⭐ 中级）

**目标**：理解浮点精度差异，学会编写数值一致性测试。

### 运行现有测试

```bash
# 需要 macOS + Apple Silicon + metal feature
cargo test --features metal --test metal_parity
```

### 理解测试结构

项目中的 `tests/metal_parity.rs` 展示了如何验证 CPU 和 GPU 的数值一致性：

```rust
// 1. 创建相同的输入数据
let input = make_vec(seed, dim, 1.0);

// 2. 分别用 CPU 和 Metal 后端计算
let cpu_result = cpu_backend.matmul_vec(&tensor, &input);
let metal_result = metal_backend.matmul_vec(&tensor, &input);

// 3. 比较结果（允许小误差）
for (c, m) in cpu_result.iter().zip(metal_result.iter()) {
    assert!((c - m).abs() < 1e-4, "mismatch: cpu={}, metal={}", c, m);
}
```

### 挑战：添加新的一致性测试

为 `rms_norm`、`softmax`、`silu` 等算子编写类似的一致性测试。

---

## 9.7 项目六：实现简单的聊天模板（⭐ 入门）

**目标**：理解 chat 模型的提示格式。

### 背景

Chat 模型（如 TinyLlama-Chat）期望特定的输入格式：

```
<|system|>
You are a helpful assistant.</s>
<|user|>
What is the capital of France?</s>
<|assistant|>
```

### 实现

在 `main.rs` 中添加 `--chat` 模式：

```rust
fn format_chat_prompt(system: &str, user: &str) -> String {
    format!(
        "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
        system, user
    )
}

// 在 parse_args 中添加 --chat 和 --system 参数
// 如果 --chat 模式，自动包装 prompt
```

**注意**：不同模型使用不同的聊天模板。TinyLlama-Chat 使用上述格式，其他模型可能不同。

---

## 9.8 扩展思路

以下是更多可以探索的方向：

### 短期项目（1-2 天）

| 项目 | 难度 | 描述 |
|------|------|------|
| 添加 `--verbose` 模式 | ⭐ | 输出每步的 token ID、概率、耗时 |
| 实现 min-p 采样 | ⭐ | 新的采样策略：保留概率 >= min_p * max_prob 的 token |
| 添加 token 计数统计 | ⭐ | 输出 prompt tokens 数、生成 tokens 数、tokens/sec |
| JSON 输出模式 | ⭐ | `--json` 输出结构化结果（token 列表、耗时等） |

### 中期项目（3-7 天）

| 项目 | 难度 | 描述 |
|------|------|------|
| 支持 Q5_1 量化 | ⭐⭐ | 类似 Q5_0 但有最小值偏移 |
| 实现 KV 缓存量化 | ⭐⭐⭐ | 将 KV 缓存从 f32 压缩为 f16，减少内存 |
| 多轮对话 | ⭐⭐ | 保持 KV 缓存，支持连续对话 |
| 流式 HTTP API | ⭐⭐ | 用 SSE 实现类似 OpenAI API 的流式接口 |

### 长期项目（1-4 周）

| 项目 | 难度 | 描述 |
|------|------|------|
| 支持 DeepSeek/Qwen 架构 | ⭐⭐⭐ | 滑动窗口注意力、不同的 FFN 结构 |
| 投机解码 | ⭐⭐⭐⭐ | 用小模型草稿 + 大模型验证加速生成 |
| 连续批处理 | ⭐⭐⭐⭐ | 同时处理多个请求，动态调度 |
| WebGPU 后端 | ⭐⭐⭐⭐ | 跨平台 GPU 加速，可在浏览器中运行 |

---

## 9.9 调试指南

### 常见问题

**1. 输出乱码**
- 检查温度是否太高（> 1.5）
- 检查模型文件是否完整下载
- 尝试 `-t 0.0` 贪婪解码确认模型正常

**2. 数值溢出（NaN/Inf）**
- 在 `rms_norm` 中检查 epsilon 是否正确
- 在 `softmax` 中确认减去了最大值
- 检查量化反量化是否正确

**3. 生成重复内容**
- 这是低温度 + 小模型的常见现象
- 尝试提高温度或降低 top-p
- 实现重复惩罚（项目一的挑战）

**4. Metal 后端崩溃**
- 确认使用 `--features metal` 编译
- 检查 GPU 层数是否超过模型层数
- 运行 `cargo test --features metal --test metal_parity` 验证

### 有用的调试命令

```bash
# 查看模型信息（元数据）
# 可以在 gguf.rs 中添加一个 dump 功能，或使用 Python 的 gguf 库
pip install gguf
python -c "
import gguf
reader = gguf.GGUFReader('model.gguf')
for k, v in reader.fields.items():
    print(f'{k}: {v.data}')
"

# 对比两次运行的输出（应该完全相同）
diff <(./target/release/mini-rsllm model.gguf -p "Hello" -n 20 -t 0.0) \
     <(./target/release/mini-rsllm model.gguf -p "Hello" -n 20 -t 0.0)

# 检查内存使用
/usr/bin/time -l ./target/release/mini-rsllm model.gguf -p "Hello" -n 50 -t 0.0
```

---

## 9.10 小结

✅ 本章提供了：

- [ ] 6 个由浅入深的实践项目
- [ ] 采样策略实验方法
- [ ] 量化格式扩展指南
- [ ] 性能分析和优化技巧
- [ ] 数值一致性测试方法
- [ ] 丰富的扩展方向

### 学习建议

1. **从简单开始**：先完成项目一（采样实验），建立信心
2. **选择感兴趣的方向**：不需要全部完成，选 1-2 个深入
3. **对照源码**：每个修改都要理解它在整体架构中的位置
4. **写测试**：每个改动都应该有对应的测试验证
5. **参考 llama.cpp**：遇到困难时，可以参考 llama.cpp 的实现

**恭喜你完成了 mini-rsllm 教学文档的学习！** 🎉

你现在应该对 LLM 推理引擎的完整实现有了深入的理解。从 GGUF 文件格式到 Transformer 架构，从量化技术到 GPU 加速，这些知识将帮助你理解和改进任何 LLM 推理系统。

**附录**：[术语表](./appendix-glossary.md) —— 快速查阅本教程中出现的所有专业术语。