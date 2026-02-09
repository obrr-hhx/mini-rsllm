# 第 5 章：LLaMA 模型实现 ⭐

> **学习目标**：理解 LLaMA 模型的完整前向传播过程，掌握配置加载、权重引用、KV 缓存和注意力计算的实现。
>
> **对应源码**：`src/model.rs`（396 行）
>
> **预计时间**：6-8 小时（核心章节）

---

## 5.1 模型配置

### LlamaConfig 结构

```rust
pub struct LlamaConfig {
    pub dim: usize,            // 隐藏层维度（如 2048）
    pub n_layers: usize,       // Transformer 层数（如 22）
    pub n_heads: usize,        // Q 注意力头数（如 32）
    pub n_kv_heads: usize,     // KV 注意力头数（如 4，GQA）
    pub vocab_size: usize,     // 词表大小（如 32000）
    pub ff_dim: usize,         // FFN 中间维度（如 5632）
    pub norm_eps: f32,         // RMS Norm 的 epsilon（如 1e-5）
    pub rope_freq_base: f32,   // RoPE 频率基数（如 10000.0）
    pub max_seq_len: usize,    // 最大序列长度（如 2048）
    pub head_dim: usize,       // 每个头的维度 = dim / n_heads（如 64）
}
```

所有配置都从 GGUF 元数据中读取：

```rust
impl LlamaConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let dim = gguf.get_u32("llama.embedding_length")
            .unwrap_or(gguf.get_u32("gpt2.embedding_length").unwrap_or(512)) as usize;
        let n_layers = gguf.get_u32("llama.block_count")
            .unwrap_or(gguf.get_u32("gpt2.block_count").unwrap_or(4)) as usize;
        // ... 其他字段类似
        let head_dim = dim / n_heads;
        // ...
    }
}
```

> **兼容性设计**：代码同时尝试 `llama.*` 和 `gpt2.*` 前缀，支持不同模型架构的 GGUF 文件。

### TinyLlama-1.1B 的典型配置

| 参数 | 值 | 含义 |
|------|-----|------|
| dim | 2048 | 每个 token 用 2048 维向量表示 |
| n_layers | 22 | 22 层 Transformer |
| n_heads | 32 | 32 个 Q 头 |
| n_kv_heads | 4 | 4 个 KV 头（GQA 比例 8:1） |
| head_dim | 64 | 2048/32 = 64 |
| ff_dim | 5632 | FFN 中间层约 2.75× dim |
| vocab_size | 32000 | 32K 词表 |

---

## 5.2 权重加载

### 权重结构

```rust
/// 单层的权重引用
pub struct LayerWeights<'a> {
    pub attn_norm: TensorRef<'a>,  // 注意力归一化 [dim]
    pub wq: TensorRef<'a>,         // Q 投影 [dim, dim]
    pub wk: TensorRef<'a>,         // K 投影 [dim, kv_dim]
    pub wv: TensorRef<'a>,         // V 投影 [dim, kv_dim]
    pub wo: TensorRef<'a>,         // 输出投影 [dim, dim]
    pub ffn_norm: TensorRef<'a>,   // FFN 归一化 [dim]
    pub w1: TensorRef<'a>,         // FFN gate [dim, ff_dim]
    pub w2: TensorRef<'a>,         // FFN down [ff_dim, dim]
    pub w3: TensorRef<'a>,         // FFN up [dim, ff_dim]
}

/// 全部权重
pub struct LlamaWeights<'a> {
    pub token_embd: TensorRef<'a>,   // 词嵌入 [dim, vocab_size]
    pub output_norm: TensorRef<'a>,  // 最终归一化 [dim]
    pub output: TensorRef<'a>,       // 输出投影 [dim, vocab_size]
    pub layers: Vec<LayerWeights<'a>>,
}
```

### 零拷贝加载

```rust
fn make_ref<'a>(gguf: &'a GgufFile, name: &str) -> TensorRef<'a> {
    let info = gguf.tensors.get(name)
        .unwrap_or_else(|| panic!("missing tensor: {}", name))
        .clone();
    let data = gguf.tensor_data(&info);  // 返回 &[u8]，指向 mmap
    TensorRef { data, info }
}
```

**关键**：`TensorRef` 只存储指向 mmap 的引用，不拷贝任何权重数据。整个模型加载几乎是瞬时的。

### 权重共享（Tied Weights）

```rust
let output = if gguf.tensors.contains_key("output.weight") {
    make_ref(gguf, "output.weight")
} else {
    make_ref(gguf, "token_embd.weight")  // 共享词嵌入作为输出投影
};
```

某些模型没有独立的输出投影矩阵，而是复用词嵌入矩阵。这节省了大量参数。

---

## 5.3 KV 缓存

```rust
pub struct KvCache {
    pub key_cache: Vec<Vec<f32>>,   // [n_layers][max_seq_len * kv_dim]
    pub val_cache: Vec<Vec<f32>>,   // [n_layers][max_seq_len * kv_dim]
    kv_dim: usize,                  // n_kv_heads * head_dim
    max_seq_len: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self {
        let size = max_seq_len * kv_dim;
        KvCache {
            key_cache: (0..n_layers).map(|_| vec![0.0; size]).collect(),
            val_cache: (0..n_layers).map(|_| vec![0.0; size]).collect(),
            kv_dim,
            max_seq_len,
        }
    }
}
```

KV 缓存的内存布局：

```
key_cache[layer_idx]:
┌──────────┬──────────┬──────────┬─────┬──────────┐
│ pos=0    │ pos=1    │ pos=2    │ ... │ pos=max  │
│ [kv_dim] │ [kv_dim] │ [kv_dim] │     │ [kv_dim] │
└──────────┴──────────┴──────────┴─────┴──────────┘
  kv_dim = n_kv_heads × head_dim = 4 × 64 = 256
```


---

## 5.4 前向传播：完整流程

这是整个项目的核心——`forward()` 方法。让我们逐步解读：

```rust
pub fn forward(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>, String> {
```

输入一个 token ID 和位置，输出 logits 向量 `[vocab_size]`。

### 第 1 步：词嵌入查表

```rust
    // 1. Embedding lookup: x = token_embd[token_id] → [dim]
    let embd_row = w.token_embd.dequantize_row(token_id as usize);
    let mut x = embd_row;
```

从词嵌入矩阵中取出 token 对应的行，得到 `[dim]` 维向量。这就是 token 的初始表示。

### 第 2 步：Transformer 层循环

```rust
    for layer_idx in 0..cfg.n_layers {
        let layer = &w.layers[layer_idx];
        let use_gpu = self.backend.should_use_layer_gpu(layer_idx, cfg.n_layers);
```

逐层处理，每层可以选择 CPU 或 GPU 执行。

#### 2a. 注意力归一化

```rust
        let attn_norm_w = dequantize(layer.attn_norm.data, ...);
        let h = backend.rms_norm(&x, &attn_norm_w, cfg.norm_eps);
```

Pre-Norm：先归一化，再进入注意力子层。`h` 是归一化后的向量。

#### 2b. Q/K/V 投影

```rust
        let mut q = backend.matmul_vec(&layer.wq, &h);  // [dim] → [dim]
        let mut k = backend.matmul_vec(&layer.wk, &h);  // [dim] → [kv_dim]
        let v = backend.matmul_vec(&layer.wv, &h);       // [dim] → [kv_dim]
```

三次矩阵-向量乘法，将归一化后的向量投影到 Q、K、V 空间。

#### 2c. 旋转位置编码

```rust
        backend.rope_qk(&mut q, &mut k, pos, head_dim, cfg.rope_freq_base);
```

对 Q 和 K 应用 RoPE，注入位置信息。注意 V 不需要位置编码。

#### 2d. 存入 KV 缓存

```rust
        let kv_offset = pos * kv_dim;
        self.kv_cache.key_cache[layer_idx][kv_offset..kv_offset + kv_dim]
            .copy_from_slice(&k);
        self.kv_cache.val_cache[layer_idx][kv_offset..kv_offset + kv_dim]
            .copy_from_slice(&v);
```

将当前位置的 K 和 V 存入缓存，供后续位置使用。

#### 2e. 多头注意力（GQA）

```rust
        let seq_len = pos + 1;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_out = backend.attention_layer(
            layer_idx, &q,
            &self.kv_cache.key_cache[layer_idx],
            &self.kv_cache.val_cache[layer_idx],
            cfg.n_heads, n_heads_per_kv, head_dim,
            seq_len, kv_dim, scale,
        );
```

这是最复杂的部分。对于每个 Q 头：
1. 找到对应的 KV 头（GQA 映射）
2. 计算 Q 与所有缓存 K 的点积 → 注意力分数
3. 缩放并 softmax → 注意力权重
4. 用权重对 V 加权求和 → 注意力输出

#### 2f. 输出投影 + 残差连接

```rust
        let attn_proj = backend.matmul_vec(&layer.wo, &attn_out);
```

#### 2g. FFN 归一化 + 残差

```rust
        x = backend.add(&x, &attn_proj);  // 残差连接
        let h = backend.rms_norm(&x, &ffn_norm_w, cfg.norm_eps);
```

#### 2h. SwiGLU FFN

```rust
        let gate = backend.matmul_vec(&layer.w1, &h);     // gate 投影
        let up = backend.matmul_vec(&layer.w3, &h);       // up 投影
        let gate_act = backend.silu(&gate);                // SiLU 激活
        let ffn_hidden = backend.mul(&gate_act, &up);      // 门控乘法
        let ffn_out = backend.matmul_vec(&layer.w2, &ffn_hidden); // down 投影
        x = backend.add(&x, &ffn_out);                    // 残差连接
```

### 第 3 步：最终归一化

```rust
    let output_norm_w = dequantize(w.output_norm.data, ...);
    x = backend.rms_norm(&x, &output_norm_w, cfg.norm_eps);
```

### 第 4 步：输出投影

```rust
    let logits = backend.matmul_vec(&w.output, &x);  // [dim] → [vocab_size]
    Ok(logits)
```

将 `[dim]` 维向量投影到 `[vocab_size]` 维，得到每个词的"得分"（logits）。

---

## 5.5 数据流可视化

一次 `forward(token_id=15043, pos=1)` 的完整数据流：

```
token_id = 15043 ("▁Hello")
    │
    ▼ 词嵌入查表
x = [0.12, -0.34, 0.56, ...] (dim=2048)
    │
    ▼ ═══════ 第 0 层 ═══════
    │
    ├─ RMS Norm → h
    ├─ Q = Wq·h → [2048]  (32 个头 × 64 维)
    ├─ K = Wk·h → [256]   (4 个头 × 64 维)
    ├─ V = Wv·h → [256]   (4 个头 × 64 维)
    ├─ RoPE(Q, K, pos=1)
    ├─ 缓存 K[0][1], V[0][1]
    ├─ 注意力: Q 与 K[0][0..1] 点积 → softmax → 加权 V
    ├─ Wo·attn → [2048]
    ├─ x = x + attn_proj  (残差)
    ├─ RMS Norm → h
    ├─ gate = SiLU(W1·h) → [5632]
    ├─ up = W3·h → [5632]
    ├─ ffn = W2·(gate⊙up) → [2048]
    └─ x = x + ffn  (残差)
    │
    ▼ ═══════ 第 1~21 层 ═══════ (同上)
    │
    ▼ 最终 RMS Norm
    │
    ▼ 输出投影: Wout·x → [32000]
    │
logits = [1.2, -0.5, 3.4, ...] (vocab_size=32000)
```

---

## 5.6 计算量分析

以 TinyLlama-1.1B 为例，单个 token 的前向传播：

| 操作 | 矩阵大小 | 浮点运算 | 次数/层 |
|------|----------|---------|---------|
| Q 投影 | [2048, 2048] | 8.4M | 1 |
| K 投影 | [2048, 256] | 1.0M | 1 |
| V 投影 | [2048, 256] | 1.0M | 1 |
| 输出投影 | [2048, 2048] | 8.4M | 1 |
| FFN gate | [2048, 5632] | 23.1M | 1 |
| FFN up | [2048, 5632] | 23.1M | 1 |
| FFN down | [5632, 2048] | 23.1M | 1 |
| **每层合计** | | **~88M** | |
| **22 层合计** | | **~1.9G** | |

加上词嵌入和输出投影，总计约 **2B FLOPs/token**。

---

## 5.7 小结

✅ 本章你学到了：

- [ ] LlamaConfig：从 GGUF 元数据读取模型配置
- [ ] LlamaWeights：零拷贝引用 mmap 中的权重数据
- [ ] KvCache：预分配的 KV 缓存，按位置存储
- [ ] forward() 的完整流程：嵌入 → N 层(注意力+FFN) → 归一化 → 投影
- [ ] GQA 注意力：Q 头多、KV 头少的分组查询
- [ ] 后端抽象：每个操作可选 CPU 或 GPU 执行

### 关键设计决策

1. **为什么权重用 TensorRef 而不是 Tensor？** TensorRef 是零拷贝引用，指向 mmap。如果用 Tensor（f32），加载 1.1B 模型需要 4.4GB 内存。
2. **为什么 KV 缓存预分配？** 避免生成过程中的动态内存分配，保证性能稳定。
3. **为什么每个操作都有 CPU/GPU 分支？** 支持层级卸载——部分层在 GPU，部分在 CPU，适应不同显存大小。

**下一章**：[第 6 章：采样策略](./06-sampling-strategies.md) —— 理解如何从 logits 中选择下一个 token。