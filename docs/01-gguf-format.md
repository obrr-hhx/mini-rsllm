# 第 1 章：GGUF 文件格式深入解析

> **学习目标**：理解 GGUF 二进制格式的结构，掌握内存映射原理，能够解析 GGUF 文件中的元数据和张量信息。
>
> **对应源码**：`src/gguf.rs`（295 行）
>
> **预计时间**：3-4 小时

---

## 1.1 什么是 GGUF？

GGUF（**G**GPT-**G**enerated **U**nified **F**ormat）是 [ggml](https://github.com/ggerganov/ggml) 项目定义的模型文件格式。它的设计目标是：

1. **自包含**：一个文件包含模型运行所需的一切——权重、配置、分词器
2. **高效加载**：支持内存映射（mmap），无需将整个文件读入内存
3. **量化友好**：原生支持多种量化格式（Q4_0、Q8_0、Q6_K 等）
4. **版本兼容**：通过版本号保证向前兼容

### GGUF 文件的整体结构

```
┌──────────────────────────────────────┐
│           Header (固定大小)           │
│  magic(4B) + version(4B)             │
│  + tensor_count(8B)                  │
│  + metadata_kv_count(8B)             │
├──────────────────────────────────────┤
│        Metadata KV Pairs             │
│  模型配置、分词器数据等               │
│  (变长，逐个读取)                     │
├──────────────────────────────────────┤
│         Tensor Info Array            │
│  每个张量的名称、形状、类型、偏移量    │
│  (变长，逐个读取)                     │
├──────────────────────────────────────┤
│         Padding (对齐到 32 字节)      │
├──────────────────────────────────────┤
│         Tensor Data (二进制)          │
│  实际的权重数据                       │
│  (通过 mmap 按需访问，不全部加载)     │
└──────────────────────────────────────┘
```

---

## 1.2 代码解读：文件打开与头部解析

让我们逐段阅读 `src/gguf.rs` 中的 `GgufFile::open()` 方法。

### 核心数据结构

```rust
// src/gguf.rs

/// 解析后的 GGUF 文件
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,  // 元数据键值对
    pub tensors: HashMap<String, TensorInfo>,  // 张量信息表
    mmap: Mmap,                                // 内存映射的文件
    data_offset: usize,                        // 张量数据区的起始偏移
}
```

这四个字段就是 GGUF 文件解析后的全部内容：
- `metadata`：存储模型配置（层数、维度等）和分词器数据（词表、合并规则等）
- `tensors`：存储每个权重张量的元信息（名称、形状、数据类型、在文件中的偏移量）
- `mmap`：内存映射对象，操作系统按需将文件页加载到内存
- `data_offset`：权重数据区在文件中的起始位置

### 第一步：内存映射

```rust
pub fn open(path: &str) -> io::Result<Self> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = Cursor::new(&mmap[..]);
    // ...
}
```

**关键概念：内存映射（Memory-Mapped I/O）**

普通文件读取 vs 内存映射：

```
普通读取：                          内存映射：
┌──────┐    read()    ┌──────┐    ┌──────┐   mmap()   ┌──────┐
│ 磁盘 │ ──────────→ │ 内存 │    │ 磁盘 │ ─────────→ │ 虚拟 │
│ 文件 │   拷贝全部   │ 缓冲 │    │ 文件 │  建立映射   │ 地址 │
└──────┘             └──────┘    └──────┘   按需加载   └──────┘
```

内存映射的优势：
- **不拷贝**：不需要将 600MB 的模型文件全部读入内存
- **按需加载**：操作系统只在访问时才将对应的文件页加载到物理内存
- **零拷贝访问**：直接通过指针访问文件内容，就像访问内存数组一样

这就是为什么 `mmap` 后可以用 `&mmap[..]` 当作字节切片来使用。

### 第二步：读取文件头

```rust
// 读取 magic number：必须是 "GGUF" 的小端编码 0x46554747
let magic = cursor.read_u32::<LittleEndian>()?;
if magic != GGUF_MAGIC {
    return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("invalid GGUF magic: 0x{:08X}", magic),
    ));
}

// 版本号：支持 v2 和 v3
let version = cursor.read_u32::<LittleEndian>()?;

// 文件中包含的张量数量和元数据键值对数量
let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;
```

文件头共 24 字节：

```
偏移  大小  字段
0     4B    magic = 0x46554747 ("GGUF")
4     4B    version (2 或 3)
8     8B    tensor_count (张量数量)
16    8B    metadata_kv_count (元数据数量)
```

> **为什么用小端序（Little-Endian）？** 因为 x86/ARM 架构都是小端序，直接读取无需转换，效率最高。

---

## 1.3 元数据解析

### 值类型系统

GGUF 定义了 13 种值类型：

```rust
pub enum GgufValue {
    U8(u8),           // 类型 0
    I8(i8),           // 类型 1
    U16(u16),         // 类型 2
    I16(i16),         // 类型 3
    U32(u32),         // 类型 4
    I32(i32),         // 类型 5
    F32(f32),         // 类型 6
    Bool(bool),       // 类型 7
```

注意 `Array` 类型是递归的——数组的元素可以是任意类型（包括嵌套数组）。

### 键值对读取

每个元数据条目的二进制布局：

```
[key_length: u64] [key_bytes: u8 * key_length] [value_type: u32] [value_data: ...]
```

对应的读取代码：

```rust
// 读取所有元数据键值对
let mut metadata = HashMap::new();
for _ in 0..metadata_kv_count {
    let key = read_string(&mut cursor)?;    // 读取键名
    let value = read_value(&mut cursor)?;   // 读取值（先读类型，再读数据）
    metadata.insert(key, value);
}
```

其中 `read_string` 的实现：

```rust
fn read_string(cursor: &mut Cursor<&[u8]>) -> io::Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;  // 字符串长度
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;                            // 读取字节
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
```

### 模型中的典型元数据

以 TinyLlama-1.1B 为例，GGUF 文件中存储的关键元数据包括：

| 键名 | 类型 | 示例值 | 用途 |
|------|------|--------|------|
| `llama.embedding_length` | U32 | 2048 | 隐藏层维度 |
| `llama.block_count` | U32 | 22 | Transformer 层数 |
| `llama.attention.head_count` | U32 | 32 | 注意力头数 |
| `llama.attention.head_count_kv` | U32 | 4 | KV 头数（GQA） |
| `llama.feed_forward_length` | U32 | 5632 | FFN 中间维度 |
| `llama.context_length` | U32 | 2048 | 最大上下文长度 |
| `llama.rope.freq_base` | F32 | 10000.0 | RoPE 频率基数 |
| `tokenizer.ggml.tokens` | Array[Str] | ["<unk>", "<s>", ...] | 词表 |
| `tokenizer.ggml.scores` | Array[F32] | [-1000.0, ...] | BPE 合并分数 |
| `tokenizer.ggml.merges` | Array[Str] | ["▁ t", ...] | BPE 合并规则 |

> **重要洞察**：GGUF 文件是自包含的——模型配置和分词器数据都在元数据中，不需要额外的配置文件。

---

## 1.4 张量信息解析

### 张量信息结构

```rust
pub struct TensorInfo {
    pub name: String,       // 张量名称，如 "blk.0.attn_q.weight"
    pub shape: Vec<usize>,  // 形状，如 [2048, 2048]
    pub dtype: GgufDType,   // 数据类型，如 Q4_0
    pub offset: usize,      // 相对于数据区起始的偏移量
}
```

### 支持的数据类型

```rust
pub enum GgufDType {
    F32 = 0,    // 全精度浮点
    F16 = 1,    // 半精度浮点
    Q4_0 = 2,   // 4 位量化
    Q8_0 = 8,   // 8 位量化
    Q6_K = 14,  // 6 位 k-quant
}
```

每种类型的存储效率：

```rust
pub fn block_size(self) -> (usize, usize) {
    match self {
        //                    (字节/块, 元素/块)
        GgufDType::F32  => (4, 1),      // 4 字节/元素
        GgufDType::F16  => (2, 1),      // 2 字节/元素
        GgufDType::Q4_0 => (18, 32),    // 0.5625 字节/元素
        GgufDType::Q8_0 => (34, 32),    // 1.0625 字节/元素
        GgufDType::Q6_K => (210, 256),  // 0.8203 字节/元素
    }
}
```

以 Q4_0 为例：每 32 个参数只需 18 字节（2 字节缩放因子 + 16 字节存 32 个 4 位值），相比 F32 的 128 字节，压缩了 **7 倍**。

### 张量信息读取

```rust
let mut tensors = HashMap::new();
for _ in 0..tensor_count {
    let name = read_string(&mut cursor)?;
    let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        shape.push(cursor.read_u64::<LittleEndian>()? as usize);
    }
    let dtype = GgufDType::from_u32(cursor.read_u32::<LittleEndian>()?)?;
    let offset = cursor.read_u64::<LittleEndian>()? as usize;

    tensors.insert(name.clone(), TensorInfo { name, shape, dtype, offset });
}
```

每个张量信息条目的二进制布局：

```
[name_length: u64] [name: u8*N] [n_dims: u32] [dim_0: u64] ... [dim_n: u64] [dtype: u32] [offset: u64]
```

### 数据区对齐

```rust
let pos = cursor.position() as usize;
let data_offset = (pos + 31) & !31;  // 对齐到 32 字节边界
```

`(pos + 31) & !31` 是一个经典的对齐技巧：将 `pos` 向上取整到最近的 32 的倍数。

例如：`pos = 100` → `(100 + 31) & !31` = `131 & 0xFFFFFFE0` = `128`

---

## 1.5 访问张量数据

解析完成后，通过 `tensor_data()` 方法获取某个张量的原始字节：

```rust
pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
    let start = self.data_offset + info.offset;
    let end = start + info.byte_size();
    &self.mmap[start..end]
}
```

这里返回的是 `&[u8]`——一个指向 mmap 区域的字节切片。**没有任何数据拷贝**，只是计算了起止位置。

张量的字节大小计算：

```rust
impl TensorInfo {
    pub fn byte_size(&self) -> usize {
        let n = self.n_elements();                       // 总元素数
        let (block_bytes, block_elems) = self.dtype.block_size();
        (n / block_elems) * block_bytes                  // 总块数 × 每块字节数
    }
}
```

### LLaMA 模型中的典型张量

| 张量名称 | 形状 | 说明 |
|----------|------|------|
| `token_embd.weight` | [dim, vocab_size] | 词嵌入矩阵 |
| `blk.{i}.attn_norm.weight` | [dim] | 注意力层归一化权重 |
| `blk.{i}.attn_q.weight` | [dim, dim] | Q 投影矩阵 |
| `blk.{i}.attn_k.weight` | [dim, kv_dim] | K 投影矩阵 |
| `blk.{i}.attn_v.weight` | [dim, kv_dim] | V 投影矩阵 |
| `blk.{i}.attn_output.weight` | [dim, dim] | 注意力输出投影 |
| `blk.{i}.ffn_norm.weight` | [dim] | FFN 归一化权重 |
| `blk.{i}.ffn_gate.weight` | [dim, ff_dim] | FFN 门控矩阵 (W1) |
| `blk.{i}.ffn_up.weight` | [dim, ff_dim] | FFN 上投影 (W3) |
| `blk.{i}.ffn_down.weight` | [ff_dim, dim] | FFN 下投影 (W2) |
| `output_norm.weight` | [dim] | 最终归一化权重 |
| `output.weight` | [dim, vocab_size] | 输出投影矩阵 |

其中 `{i}` 是层索引（0 到 n_layers-1）。

---

## 1.6 小结

✅ 本章你学到了：

- [ ] GGUF 文件的四段式结构：Header → Metadata → Tensor Info → Tensor Data
- [ ] 内存映射（mmap）的原理和优势：按需加载，零拷贝访问
- [ ] 元数据系统：13 种值类型，存储模型配置和分词器数据
- [ ] 张量信息：名称、形状、量化类型、偏移量
- [ ] 数据对齐：32 字节边界对齐的技巧

### 关键设计决策

1. **为什么用 mmap？** 模型文件可能有几个 GB，全部读入内存不现实。mmap 让操作系统按需加载，内存占用与实际访问量成正比。
2. **为什么元数据和张量信息分开？** 元数据是小量的配置信息，需要全部解析；张量数据是大量的权重，只在计算时按需访问。
3. **为什么需要对齐？** 32 字节对齐保证了 SIMD 指令和 GPU 可以高效访问数据。

**下一章**：[第 2 章：分词器](./02-tokenizer.md) —— 理解文本如何转换为模型能处理的 token 序列。