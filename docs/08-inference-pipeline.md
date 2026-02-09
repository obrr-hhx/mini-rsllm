# 第 8 章：完整推理流程

> **学习目标**：理解从命令行输入到文本输出的完整推理流程，掌握预填充和生成两个阶段的区别。
>
> **对应源码**：`src/main.rs`（273 行）
>
> **预计时间**：3-4 小时

---

## 8.1 全局视角

前面的章节分别讲解了各个模块。现在让我们把它们串联起来，看看一次完整的推理是如何工作的：

```
命令行参数
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. 初始化                                        │
│    加载 GGUF → 构建 Tokenizer → 构建 Backend     │
│    → 创建 Model (含 KV Cache)                    │
│    → 编码 Prompt → 创建 Sampler                  │
├─────────────────────────────────────────────────┤
│ 2. 预填充 (Prefill)                              │
│    for token in prompt_tokens:                   │
│        logits = model.forward(token, pos++)      │
│    next_token = sampler.sample(last_logits)      │
├─────────────────────────────────────────────────┤
│ 3. 生成 (Generation)                             │
│    loop:                                         │
│        print(decode(token))                      │
│        if token == EOS: break                    │
│        logits = model.forward(token, pos++)      │
│        token = sampler.sample(logits)            │
└─────────────────────────────────────────────────┘
```

---

## 8.2 命令行参数

```rust
struct Args {
    model_path: String,      // GGUF 模型文件路径
    prompt: String,          // 输入提示词
    n_tokens: usize,         // 最大生成 token 数（默认 256）
    temperature: f32,        // 采样温度（默认 0.8）
    top_p: f32,              // Top-p 采样（默认 0.9）
    top_k: usize,            // Top-k 采样（默认 40）
    seed: u64,               // 随机种子（默认 42）
    device: DeviceKind,      // 后端设备（默认 cpu）
    gpu_layers: usize,       // GPU 层数（默认 0）
}
```

使用示例：

```bash
mini-rsllm model.gguf -p "Once upon a time" -n 100 -t 0.8 --top-k 40 --top-p 0.9
```

### 参数验证

```rust
// 温度不能为负
if result.temperature < 0.0 { exit(1); }
// top-p 必须在 (0, 1] 范围
if !(0.0 < result.top_p && result.top_p <= 1.0) { exit(1); }
// CPU 模式下不能指定 GPU 层数
if result.device == DeviceKind::Cpu && result.gpu_layers > 0 { exit(1); }
```

---

## 8.3 初始化阶段

```rust
fn main() {
    let args = parse_args();

    // 1. 加载 GGUF 文件（内存映射）
    let gguf = GgufFile::open(&args.model_path).expect("failed to open GGUF file");

    // 2. 构建分词器
    let tokenizer = Tokenizer::from_gguf(&gguf);

    // 3. 构建后端
    let backend = build_backend(args.device, args.gpu_layers)?;

    // 4. 创建模型（含 KV 缓存）
    let mut model = LlamaModel::from_gguf_with_backend(&gguf, backend);

    // 5. 编码提示词
    let prompt_tokens = tokenizer.encode(&args.prompt, true);  // true = 添加 BOS

    // 6. 创建采样器
    let mut sampler = Sampler::new(SamplerConfig { ... });
}
```

**初始化顺序很重要**：
- GGUF 必须先加载（其他模块都依赖它）
- Tokenizer 和 Model 都从 GGUF 构建
- Backend 在 Model 之前创建（Model 需要 Backend）

### 生成长度限制

```rust
let max_new_tokens = args.n_tokens
    .min(model.config.max_seq_len.saturating_sub(prompt_tokens.len()));
```

确保 prompt + 生成的 token 总数不超过模型的上下文长度。

---

## 8.4 预填充阶段（Prefill）

预填充处理提示词中的所有 token，填充 KV 缓存：

```rust
for i in 0..prompt_tokens.len() {
    let mut logits = model.forward(prompt_tokens[i], pos).unwrap();
    pos += 1;

    if i == prompt_tokens.len() - 1 {
        // 只在最后一个 prompt token 时采样
        token = sampler.sample(&mut logits);
    }
}
```

### 为什么需要预填充？

```
Prompt: "The cat sat on the"
Tokens: [1, 450, 6635, 3290, 373, 278]

预填充过程：
  pos=0: forward(1)      → logits (丢弃)     KV[0] 已填充
  pos=1: forward(450)    → logits (丢弃)     KV[1] 已填充
  pos=2: forward(6635)   → logits (丢弃)     KV[2] 已填充
  pos=3: forward(3290)   → logits (丢弃)     KV[3] 已填充
  pos=4: forward(373)    → logits (丢弃)     KV[4] 已填充
  pos=5: forward(278)    → logits → 采样     KV[5] 已填充
                                    ↓
                              next_token = "mat"
```

**关键点**：
- 每次 forward 都会更新 KV 缓存
- 只有最后一个 token 的 logits 用于采样（因为我们要预测的是 prompt 之后的下一个词）
- 前面的 logits 被丢弃，但 KV 缓存保留了所有信息

---

## 8.5 生成阶段（Generation）

预填充完成后，进入自回归生成循环：

```rust
gen_start = Instant::now();
for _step in 0..max_new_tokens {
    // 1. 解码并输出当前 token
    let text = tokenizer.decode_token(token);
    print!("{}", text);
    io::stdout().flush().unwrap();

    // 2. 检查是否遇到结束符
    if token == tokenizer.eos_id {
        break;
    }

    // 3. 前向传播：用当前 token 预测下一个
    let mut logits = model.forward(token, pos).unwrap();
    pos += 1;
    n_generated += 1;

    // 4. 采样下一个 token
    token = sampler.sample(&mut logits);
}
println!();  // 最后换行
```

### 自回归循环详解

```
预填充结束后：token = "mat", pos = 6

Step 0:
  输出 "mat"
  forward("mat", pos=6) → logits → 采样 → "ress"
  KV[6] 已填充, pos = 7

Step 1:
  输出 "ress"
  forward("ress", pos=7) → logits → 采样 → " on"
  KV[7] 已填充, pos = 8

Step 2:
  输出 " on"
  forward(" on", pos=8) → logits → 采样 → " the"
  KV[8] 已填充, pos = 9

...直到遇到 EOS 或达到 max_new_tokens
```

### 循环结构的关键设计

**先输出，后计算**：注意循环中先 `print` 再 `forward`。这是因为：
1. 预填充阶段已经采样了第一个生成 token
2. 先输出这个 token，再用它计算下一个
3. 这样 EOS token 也会被检测到但不会触发多余的 forward

**EOS 检测位置**：在 `forward` 之前检查 EOS，避免浪费一次前向传播。

---

## 8.6 流式输出

### 为什么需要流式输出？

LLM 生成是逐 token 进行的，每个 token 需要一次完整的前向传播。如果等所有 token 生成完再输出，用户需要等待很长时间。

**流式输出**让用户看到"打字机效果"——每生成一个 token 就立即显示：

```rust
// 关键：print! 不换行，flush 立即刷新
print!("{}", text);           // 不换行输出
io::stdout().flush().unwrap(); // 立即刷新缓冲区
```

### 为什么需要 flush？

标准输出默认是**行缓冲**的——只有遇到换行符 `\n` 才会真正输出。但 token 通常不包含换行，所以需要手动 `flush()` 强制刷新。

```
不 flush 的效果：
  [等待很久...] 然后一次性输出所有文本

flush 的效果：
  The ← 立即显示
  The cat ← 0.1s 后
  The cat sat ← 0.1s 后
  The cat sat on ← 0.1s 后
  ...
```

### Token 解码

每个 token ID 通过 `decode_token()` 转换为文本片段：

```rust
pub fn decode_token(&self, token_id: u32) -> String {
    // 1. 查找 token 对应的文本
    let piece = &self.vocab[token_id as usize];

    // 2. 处理特殊 token
    //    - 字节回退: <0x0A> → '\n'
    //    - SentencePiece 空格: '▁' → ' '

    // 3. 返回可显示的文本
}
```

---

## 8.7 性能考量

### 计时统计

`main.rs` 使用 `Instant` 记录各阶段耗时：

```rust
let total_start = Instant::now();    // 总计时
let prefill_start = Instant::now();  // 预填充计时
// ... prefill ...
let _prefill_time = prefill_start.elapsed().as_secs_f64();

gen_start = Instant::now();          // 生成计时
// ... generation ...
let _gen_time = gen_start.elapsed().as_secs_f64();
let _total_time = total_start.elapsed().as_secs_f64();
```

### 预填充 vs 生成的性能差异

| 特性 | 预填充 | 生成 |
|------|--------|------|
| 处理方式 | 逐 token 顺序处理 | 逐 token 顺序处理 |
| KV 缓存 | 从空开始填充 | 在已有缓存上追加 |
| 注意力范围 | 随 pos 增长 | 随 pos 增长 |
| 瓶颈 | 计算密集 | 内存带宽 |
| 输出 | 无（静默处理） | 每步输出一个 token |

> **注意**：本项目的预填充是逐 token 处理的（非批量），这是为了代码简洁。
> 生产级实现（如 llama.cpp）会批量处理 prompt tokens 以利用矩阵并行性。

### 内存使用模式

```
┌─────────────────────────────────────────────┐
│ 模型权重（mmap，按需加载）                      │  ← 最大部分
├─────────────────────────────────────────────┤
│ KV 缓存（预分配，固定大小）                      │  ← 第二大
│   大小 = n_layers × 2 × max_seq_len × dim   │
├─────────────────────────────────────────────┤
│ 中间缓冲区（每步复用）                          │  ← 较小
│   logits, attention scores, FFN 中间值       │
└─────────────────────────────────────────────┘
```

**关键优化**：
- **mmap**：模型权重不完整加载到内存，由 OS 按需调页
- **KV 缓存预分配**：避免生成过程中的动态分配
- **缓冲区复用**：每步的中间结果覆盖上一步

### GPU 加速的影响

```
CPU only:          ~5 tokens/sec  (小模型)
Metal (全部层):    ~20 tokens/sec (小模型)
Metal (部分层):    介于两者之间

加速比取决于：
- 模型大小（越大越受益于 GPU）
- 量化格式（Q4_0 在 GPU 上有专用 kernel）
- GPU 层数（更多层 = 更快，但需要更多显存）
```

---

## 8.8 完整流程回顾

让我们用一个具体例子串联整个流程：

```bash
mini-rsllm llama-3.2-1b-q4_0.gguf -p "Hello" -n 5 -t 0.8
```

```
第 1 步：解析参数
  model_path = "llama-3.2-1b-q4_0.gguf"
  prompt = "Hello", n_tokens = 5, temperature = 0.8

第 2 步：加载 GGUF（mmap）
  → 解析 metadata: vocab_size=32000, n_layers=22, dim=2048...
  → 解析 tensor info: 200+ 个张量的名称、形状、偏移量

第 3 步：构建 Tokenizer
  → 从 metadata 提取词汇表和合并规则
  → 编码 "Hello" → [1, 15043]  (BOS + "Hello")

第 4 步：构建 Backend + Model
  → CpuBackend（默认）
  → LlamaModel: 22 层, dim=2048, KV 缓存已分配

第 5 步：预填充
  pos=0: forward(1)     → KV[0] 填充, logits 丢弃
  pos=1: forward(15043) → KV[1] 填充, logits → 采样 → token_2

第 6 步：生成（5 个 token）
  Step 0: 输出 decode(token_2), forward → token_3
  Step 1: 输出 decode(token_3), forward → token_4
  Step 2: 输出 decode(token_4), forward → token_5
  Step 3: 输出 decode(token_5), forward → token_6
  Step 4: 输出 decode(token_6), 达到 max_new_tokens, 结束

输出: "Hello, how are you doing"
```

---

## 8.9 小结

✅ 本章你学到了：

- [ ] 完整推理流程的三个阶段：初始化 → 预填充 → 生成
- [ ] 命令行参数解析和验证
- [ ] 初始化的依赖顺序：GGUF → Tokenizer → Backend → Model → Sampler
- [ ] 预填充：处理 prompt，填充 KV 缓存，只在最后采样
- [ ] 生成：自回归循环，先输出后计算
- [ ] 流式输出：`print!` + `flush()` 实现打字机效果
- [ ] 性能特征：预填充 vs 生成的差异

### 关键设计决策

1. **为什么分预填充和生成两个阶段？** 预填充处理已知的 prompt，不需要输出；生成是逐步探索，需要实时输出。分开处理逻辑更清晰。
2. **为什么逐 token 预填充而不是批量？** 为了代码简洁。批量预填充需要修改 forward 接口支持多 token 输入，增加复杂度。
3. **为什么先输出后计算？** 预填充已经产生了第一个 token，先输出它可以减少用户感知的延迟。

**下一章**：[第 9 章：实践与扩展](./09-hands-on-projects.md) —— 动手实践项目，巩固所学知识。