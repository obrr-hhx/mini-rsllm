# 第 3 章：张量与量化

> **学习目标**：理解量化的原理和实现，掌握核心数学运算（矩阵乘法、归一化、激活函数、位置编码）。
>
> **对应源码**：`src/tensor.rs`（396 行）
>
> **预计时间**：4-5 小时

---

## 3.1 什么是量化？

LLM 的权重通常以 32 位浮点数（F32）训练，但推理时不需要这么高的精度。**量化**就是用更少的位数来表示权重，从而：

1. **减小模型体积**：7B 参数的 F32 模型约 28GB，Q4_0 量化后仅约 4GB
2. **减少内存带宽**：推理的瓶颈通常是内存带宽，数据越小读取越快
3. **保持精度**：精心设计的量化方案可以将精度损失控制在可接受范围

### 量化的基本思想

```
原始 F32 权重:  [0.123, -0.456, 0.789, -0.234, ...]
                每个值 4 字节，32个值共 128 字节

Q4_0 量化后:    scale=0.789, quants=[1, -5, 7, -2, ...]
                每个值 4 位 + 共享缩放因子，32个值只需共 18 字节
                压缩比: 128/18 ≈ 7x
```

核心公式：`原始值 ≈ scale × 量化值`

---

## 3.2 数据结构

### Tensor：运行时张量

```rust
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,    // 所有计算都在 f32 下进行
    pub shape: Vec<usize>, // 形状，如 [2048] 或 [2048, 2048]
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor { data, shape }
    }
}
```

`Tensor` 是**运行时**的中间结果容器——所有计算都在 f32 精度下进行。

### TensorRef：权重引用

```rust
pub struct TensorRef<'a> {
    pub data: &'a [u8],      // 指向 mmap 的原始字节（可能是量化的）
    pub info: TensorInfo,     // 元信息：形状、数据类型、偏移量
}
```

`TensorRef` 是**存储时**的权重引用——直接指向 GGUF 文件的 mmap 区域，数据可能是量化格式。

```
                    ┌─────────────────────────┐
  TensorRef ──────→ │  GGUF mmap (量化数据)    │
  (零拷贝引用)       │  Q4_0 / Q8_0 / F16 ...  │
                    └─────────────────────────┘
                              │
                    dequantize / dot_product
                              │
                              ▼
                    ┌─────────────────────────┐
                    │  Tensor (f32 数据)       │
                    │  用于中间计算             │
                    └─────────────────────────┘
```

---

## 3.3 反量化实现

### 统一入口

```rust
pub fn dequantize(data: &[u8], dtype: GgufDType, n_elements: usize) -> Vec<f32> {
    match dtype {
        GgufDType::F32  => dequantize_f32(data, n_elements),
        GgufDType::F16  => dequantize_f16(data, n_elements),
        GgufDType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgufDType::Q4_0 => dequantize_q4_0(data, n_elements),
        GgufDType::Q6_K => dequantize_q6_k(data, n_elements),
    }
}
```

### F32 和 F16：无量化 / 半精度

```rust
// F32：直接从字节读取，无需转换
fn dequantize_f32(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = [data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]];
        out.push(f32::from_le_bytes(bytes));
    }
    out
}

// F16：半精度浮点 → f32
fn dequantize_f16(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = [data[i*2], data[i*2+1]];
        out.push(f16::from_le_bytes(bytes).to_f32());
    }
    out
}
```

### Q8_0：8 位量化

每个块 34 字节 = 1 个 f16 缩放因子 + 32 个 i8 量化值：

```
┌──────────┬──────────────────────────────────┐
│ scale(2B)│  q[0] q[1] q[2] ... q[31] (32B) │
│   f16    │         每个 i8 (-128~127)        │
└──────────┴──────────────────────────────────┘
  共 34 字节，表示 32 个值
```

```rust
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);
    for block in 0..n_blocks {
        let offset = block * 34;
        let scale = f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
        for i in 0..32 {
            let quant = data[offset + 2 + i] as i8;
            out.push(scale * quant as f32);  // 反量化：scale × quant
        }
```

### Q4_0：4 位量化

每个块 18 字节 = 1 个 f16 缩放因子 + 16 字节（32 个 4 位值打包）：

```
┌──────────┬──────────────────────────────────┐
│ scale(2B)│  packed[0] packed[1] ... (16B)   │
│   f16    │  每字节存 2 个 4-bit 值            │
└──────────┴──────────────────────────────────┘
  共 18 字节，表示 32 个值
```

**4 位打包方式**：

```
一个字节 = 0xAB
  低 4 位 (B): 位置 0-15 的值
  高 4 位 (A): 位置 16-31 的值

值范围: 0~15，减去偏移 8 → -8~7
```

```rust
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);
    for block in 0..n_blocks {
        let offset = block * 18;
        let scale = f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
        // 前半：低 4 位（位置 0-15）
        for i in 0..16 {
            let byte = data[offset + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;  // 取低 4 位，减偏移
            out.push(scale * lo as f32);
        }
        // 后半：高 4 位（位置 16-31）
        for i in 0..16 {
            let byte = data[offset + 2 + i];
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;  // 取高 4 位，减偏移
            out.push(scale * hi as f32);
        }
    }
    out
}
```

**精度**：每个值 4 位，只能表示 16 个级别（-8 到 7）。压缩比 ≈ 7x。

### Q6_K：6 位 k-quant

Q6_K 是最复杂的量化格式，每个块 210 字节表示 256 个值：

```
┌────────────┬──────────┬──────────┬────────┐
│ ql[128B]   │ qh[64B]  │ sc[16B]  │ d(2B)  │
│ 低 4 位     │ 高 2 位   │ 子块缩放  │ 全局缩放│
└────────────┴──────────┴──────────┴────────┘
  共 210 字节，表示 256 个值
```

每个值由 6 位组成：4 位来自 `ql`，2 位来自 `qh`，然后减去偏移 32。
反量化公式：`value = d × sub_scale × (6bit_quant - 32)`

> Q6_K 的实现较复杂，建议先理解 Q4_0 和 Q8_0，再回来研究 Q6_K。

---

## 3.4 核心运算：矩阵-向量乘法

这是整个推理引擎中**最关键的运算**——Transformer 的每一层都需要多次矩阵-向量乘法。

### 逐行反量化策略

```rust
pub fn matmul_vec(mat: &TensorRef, vec: &[f32]) -> Vec<f32> {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut out = Vec::with_capacity(rows);
    let (block_bytes, block_elems) = mat.info.dtype.block_size();
    let row_bytes = (cols / block_elems) * block_bytes;

    for row in 0..rows {
        let offset = row * row_bytes;
        let row_data = &mat.data[offset..offset + row_bytes];
        let dot = dot_product_quantized(row_data, vec, mat.info.dtype, cols);
        out.push(dot);
    }
    out
}
```

**关键设计**：不是先反量化整个矩阵再做乘法，而是**逐行**处理：

```
传统方式（内存大）：                    本项目方式（内存小）：
mat[2048×2048] Q4_0                   mat[2048×2048] Q4_0
    ↓ 反量化全部                           ↓ 逐行处理
mat_f32[2048×2048] (16MB)             for each row:
    ↓ 矩阵乘法                             反量化 1 行 (8KB)
out[2048]                                  点积 → out[row]
                                           释放该行
                                      out[2048]
```

### 量化点积

更进一步，点积可以在**块级别**直接计算，避免完整反量化一行：

```rust
// Q8_0 的量化点积
GgufDType::Q8_0 => {
    let n_blocks = n / 32;
    let mut sum = 0.0f32;
    for block in 0..n_blocks {
        let bo = block * 34;
        let scale = f16::from_le_bytes([row_data[bo], row_data[bo + 1]]).to_f32();
        let vi = block * 32;
        let mut block_sum = 0.0f32;
        for i in 0..32 {
            let quant = row_data[bo + 2 + i] as i8;
            block_sum += quant as f32 * vec[vi + i];
        }
        sum += scale * block_sum;  // scale 提到块外面，减少乘法次数
    }
    sum
}
```

**优化技巧**：`scale` 是整个块共享的，所以先累加 `quant × vec[i]`，最后再乘 `scale`。这样 32 次乘法变成 33 次（32 次 quant×vec + 1 次 scale×sum），而不是 64 次。

---

## 3.5 核心运算：归一化与激活函数

### RMS Normalization

LLaMA 使用 RMS Norm 而不是 Layer Norm（更简单，没有偏置项）：

```rust
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    // 1. 计算均方根
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let rms = (ss / n as f32 + eps).sqrt();

    // 2. 归一化并乘以权重
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((x[i] / rms) * weight[i]);
    }
    out
}
```

数学公式：`RMSNorm(x)_i = (x_i / √(mean(x²) + ε)) × w_i`

### Softmax

数值稳定的 softmax 实现（减去最大值防止溢出）：

```rust
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();  // 减去 max 防止 exp 溢出
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;              // 归一化为概率分布
    }
}
```

### SiLU 激活函数

SiLU（Sigmoid Linear Unit）= x × σ(x)，用于 FFN 的门控机制：

```rust
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * (1.0 / (1.0 + (-v).exp()))).collect()
}
```

```
SiLU(x) = x × sigmoid(x) = x × 1/(1 + e^(-x))

特点：
- x > 0 时近似线性
- x < 0 时被抑制（但不完全为 0，不像 ReLU）
- 平滑可导，训练效果好
```

---

## 3.6 核心运算：旋转位置编码（RoPE）

RoPE 是 LLaMA 的位置编码方式，通过旋转向量来编码位置信息：

```rust
pub fn rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let n_q_heads = q.len() / head_dim;
    let n_k_heads = k.len() / head_dim;

    for h in 0..n_q_heads {
        let offset = h * head_dim;
        apply_rope_to_head(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
    }
    for h in 0..n_k_heads {
        let offset = h * head_dim;
        apply_rope_to_head(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
    }
}

fn apply_rope_to_head(head: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / freq_base.powf(i as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let x0 = head[i];
        let x1 = head[i + 1];
        head[i]     = x0 * cos_t - x1 * sin_t;   // 旋转变换
        head[i + 1] = x0 * sin_t + x1 * cos_t;
    }
}
```

**直觉理解**：将每对相邻元素 (x0, x1) 看作二维平面上的点，RoPE 将其旋转 θ 角度。不同维度对使用不同频率，低维度旋转快（捕捉局部位置），高维度旋转慢（捕捉全局位置）。

```
频率计算: freq_i = 1 / base^(i/dim)
  i=0:  freq = 1/10000^0 = 1.0        （旋转最快）
  i=2:  freq = 1/10000^(2/64) ≈ 0.72
  ...
  i=62: freq = 1/10000^(62/64) ≈ 0.0001 （旋转最慢）

旋转角度: θ = position × freq
  位置 0: 不旋转
  位置 1: 旋转 freq 弧度
  位置 n: 旋转 n × freq 弧度
```

---

## 3.7 辅助运算

```rust
/// 逐元素加法
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// 逐元素乘法
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}
```

---

## 3.8 小结

✅ 本章你学到了：

- [ ] 量化的基本原理：用更少的位数表示权重，通过缩放因子恢复
- [ ] 五种数据格式：F32、F16、Q8_0、Q4_0、Q6_K 的存储结构和反量化方法
- [ ] 矩阵-向量乘法的逐行策略：避免完整反量化，节省内存
- [ ] 量化点积优化：块级别计算，scale 提取到外层
- [ ] 核心数学运算：RMS Norm、Softmax、SiLU、RoPE

### 关键设计决策

1. **为什么所有计算都用 f32？** 量化只用于存储，计算时反量化为 f32 保证精度。这是精度和存储的最佳平衡。
2. **为什么逐行反量化？** 一个 2048×2048 的 F32 矩阵需要 16MB，逐行只需 8KB。对于内存受限的设备至关重要。
3. **为什么 RoPE 而不是绝对位置编码？** RoPE 可以自然地外推到训练时未见过的序列长度，且不需要额外的可学习参数。

**下一章**：[第 4 章：Transformer 基础](./04-transformer-basics.md) —— 理解注意力机制和 Transformer 架构的理论基础。