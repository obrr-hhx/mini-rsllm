# 第 4 章：Transformer 基础

> **学习目标**：理解 Transformer 架构的核心组件，为下一章的 LLaMA 模型实现打下理论基础。
>
> **本章性质**：理论章节，不直接对应单个源文件，而是为 `model.rs` 的理解做铺垫。
>
> **预计时间**：3-4 小时

---

## 4.1 从全局看 Transformer

Transformer 是一种序列到序列的神经网络架构。LLaMA 使用的是 **decoder-only** 变体——只有解码器，没有编码器。

### 整体结构

```
输入 token
    │
    ▼
┌──────────────┐
│  词嵌入层     │  token ID → 向量 [dim]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Transformer  │ ×N 层（如 22 层）
│    Block      │
│              │
│ ┌──────────┐ │
│ │ RMS Norm │ │
│ │ Attention│ │  自注意力 + KV 缓存
│ │ Residual │ │
│ ├──────────┤ │
│ │ RMS Norm │ │
│ │   FFN    │ │  前馈网络（SwiGLU）
│ │ Residual │ │
│ └──────────┘ │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  RMS Norm    │  最终归一化
│  线性投影     │  [dim] → [vocab_size]
└──────┬───────┘
       │
       ▼
   logits [vocab_size]
```

每个 Transformer Block 包含两个子层：
1. **自注意力层**（Self-Attention）：让每个位置"看到"序列中的其他位置
2. **前馈网络**（FFN）：对每个位置独立地进行非线性变换

每个子层都有 **残差连接**（Residual Connection）和 **归一化**（RMS Norm）。

---

## 4.2 自注意力机制

### 核心思想

注意力机制回答一个问题：**"在生成当前 token 时，应该关注序列中的哪些位置？"**

### QKV 三元组

对于输入向量 x，通过三个线性变换得到：
- **Q（Query）**：当前位置的"查询"——"我在找什么？"
- **K（Key）**：每个位置的"键"——"我有什么？"
- **V（Value）**：每个位置的"值"——"如果被选中，我提供什么？"

```
Q = x @ W_q    # [dim] → [dim]
K = x @ W_k    # [dim] → [kv_dim]
V = x @ W_v    # [dim] → [kv_dim]
```

### 注意力计算

```
                    Q × K^T
score = ─────────────────────
         √(head_dim)

attention_weights = softmax(score)    # 归一化为概率
output = attention_weights × V        # 加权求和
```

**直觉**：Q 和 K 的点积衡量"相关性"。点积越大，说明当前位置越应该关注那个位置。除以 √(head_dim) 是为了防止点积值过大导致 softmax 饱和。

### 多头注意力

将 Q、K、V 分成多个"头"（head），每个头独立计算注意力，最后拼接：

```
dim = 2048, n_heads = 32
head_dim = dim / n_heads = 64

Q 分成 32 个头: Q_0[64], Q_1[64], ..., Q_31[64]
K 分成 4 个头:  K_0[64], K_1[64], K_2[64], K_3[64]  (GQA)
V 分成 4 个头:  V_0[64], V_1[64], V_2[64], V_3[64]  (GQA)

每个头独立计算注意力，然后拼接 → [2048]
```

**为什么多头？** 不同的头可以关注不同类型的关系——有的关注语法，有的关注语义，有的关注位置。

---

## 4.3 分组查询注意力（GQA）

标准多头注意力中，Q、K、V 的头数相同。**GQA（Grouped-Query Attention）** 让 K 和 V 使用更少的头：

```
标准 MHA:  Q=32头, K=32头, V=32头  → 参数多
GQA:       Q=32头, K=4头,  V=4头   → 参数少，速度快

每 8 个 Q 头共享 1 个 KV 头：
Q_0~Q_7   共享 K_0, V_0
Q_8~Q_15  共享 K_1, V_1
Q_16~Q_23 共享 K_2, V_2
Q_24~Q_31 共享 K_3, V_3
```

**优势**：
- KV 缓存减小 8 倍（32/4），节省大量内存
- 推理速度更快
- 精度损失很小

---

## 4.4 KV 缓存

在自回归生成中，每次只生成一个 token，但需要看到之前所有 token 的 K 和 V。

**没有 KV 缓存**：每次生成都要重新计算所有位置的 K 和 V → O(n²) 计算量

**有 KV 缓存**：缓存之前位置的 K 和 V，每次只计算新位置的 → O(n) 计算量

```
位置 0: 计算 K_0, V_0 → 存入缓存
位置 1: 计算 K_1, V_1 → 存入缓存，用 Q_1 和 [K_0,K_1] 计算注意力
位置 2: 计算 K_2, V_2 → 存入缓存，用 Q_2 和 [K_0,K_1,K_2] 计算注意力
...
```

KV 缓存的大小：`n_layers × 2 × n_kv_heads × max_seq_len × head_dim × sizeof(f32)`

以 TinyLlama 为例：`22 × 2 × 4 × 2048 × 64 × 4 = 约 88MB`


---

## 4.5 前馈网络（SwiGLU FFN）

每个 Transformer Block 的第二个子层是前馈网络。LLaMA 使用 **SwiGLU** 变体：

```
标准 FFN:    FFN(x) = W2 · ReLU(W1 · x)
SwiGLU FFN:  FFN(x) = W2 · (SiLU(W1 · x) ⊙ W3 · x)
```

其中 `⊙` 表示逐元素乘法。

```
输入 x [dim=2048]
    │
    ├──→ W_gate · x → [ff_dim=5632] → SiLU() → gate
    │
    ├──→ W_up · x   → [ff_dim=5632] ──────────→ up
    │
    │    gate ⊙ up → [ff_dim=5632]
    │         │
    │         ▼
    │    W_down · (gate ⊙ up) → [dim=2048]
    │
    └──→ + 残差连接 → 输出 [dim=2048]
```

**为什么用 SwiGLU？**
- 门控机制（gate）让网络学会选择性地激活特征
- SiLU 比 ReLU 更平滑，梯度更好
- 实验证明 SwiGLU 在相同参数量下效果更好

---

## 4.6 残差连接与归一化

### 残差连接

```
output = x + SubLayer(Norm(x))
```

残差连接解决深层网络的梯度消失问题——即使子层学到的变换很小，信息也能通过"捷径"传递。

### Pre-Norm vs Post-Norm

```
Post-Norm (原始 Transformer):     Pre-Norm (LLaMA):
output = Norm(x + SubLayer(x))    output = x + SubLayer(Norm(x))
```

LLaMA 使用 **Pre-Norm**（先归一化再进子层），训练更稳定。

### RMS Norm vs Layer Norm

```
Layer Norm:  y = (x - mean) / std × γ + β    (有均值偏移和偏置)
RMS Norm:    y = x / RMS(x) × γ              (只有缩放，更简单)
```

RMS Norm 省去了均值计算和偏置参数，计算更快，效果相当。

---

## 4.7 位置编码：RoPE

Transformer 本身不感知位置——打乱输入顺序，输出不变。需要位置编码来注入位置信息。

### RoPE 的核心思想

**旋转位置编码**（Rotary Position Embedding）通过旋转 Q 和 K 向量来编码位置：

```
Q_pos = Rotate(Q, pos)
K_pos = Rotate(K, pos)

score = Q_pos · K_pos^T
      = Q · R(pos_q - pos_k) · K^T
```

关键性质：**两个位置的注意力分数只取决于它们的相对距离**（pos_q - pos_k），而不是绝对位置。

### 旋转的几何直觉

将向量的每对相邻维度看作二维平面上的点，按位置旋转：

```
维度 (0,1): 旋转角度 = pos × freq_0  （高频，变化快）
维度 (2,3): 旋转角度 = pos × freq_1  （较低频）
维度 (4,5): 旋转角度 = pos × freq_2  （更低频）
...
维度 (62,63): 旋转角度 = pos × freq_31 （最低频，变化慢）
```

低频维度编码长距离依赖，高频维度编码短距离依赖。

---

## 4.8 完整的 Transformer Block

把所有组件组合起来：

```
输入 x [dim]
    │
    ├──→ RMS Norm ──→ Q = W_q · norm_x
    │                  K = W_k · norm_x
    │                  V = W_v · norm_x
    │                  RoPE(Q, K, pos)
    │                  K, V → 存入 KV 缓存
    │                  attn = softmax(Q·K^T/√d) · V
    │                  out = W_o · attn
    │
    └──→ h = x + out  (残差连接)
              │
              ├──→ RMS Norm ──→ gate = SiLU(W_gate · norm_h)
              │                  up = W_up · norm_h
              │                  ffn = W_down · (gate ⊙ up)
              │
              └──→ output = h + ffn  (残差连接)
```

---

## 4.9 小结

✅ 本章你学到了：

- [ ] Transformer 的整体架构：嵌入 → N 个 Block → 归一化 → 投影
- [ ] 自注意力机制：QKV 三元组、注意力分数、加权求和
- [ ] GQA：Q 头多、KV 头少，节省内存和计算
- [ ] KV 缓存：避免重复计算，O(n²) → O(n)
- [ ] SwiGLU FFN：门控前馈网络，比标准 FFN 效果更好
- [ ] 残差连接和 Pre-Norm：保证深层网络的训练稳定性
- [ ] RoPE：通过旋转编码相对位置，支持长度外推

### 与代码的对应关系

| 概念 | 代码位置 |
|------|----------|
| 词嵌入 | `model.rs` - `token_embd.weight` 查表 |
| RMS Norm | `tensor.rs` - `rms_norm()` |
| Q/K/V 投影 | `model.rs` - `matmul_vec(attn_q/k/v)` |
| RoPE | `tensor.rs` - `rope()` |
| 注意力计算 | `model.rs` - `forward()` 中的注意力循环 |
| KV 缓存 | `model.rs` - `KvCache` 结构 |
| SwiGLU FFN | `model.rs` - gate/up/down 矩阵乘法 |
| 残差连接 | `model.rs` - `tensor::add()` |

**下一章**：[第 5 章：LLaMA 模型实现](./05-llama-model.md) —— 将理论付诸实践，逐行解读模型的前向传播代码。
