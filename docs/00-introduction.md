# 第 0 章：导读与环境搭建

> **学习目标**：理解项目定位和学习价值，搭建开发环境，成功运行第一个推理示例，了解整体架构。
>
> **预计时间**：1-2 小时

---

## 0.1 mini-rsllm 是什么？

**mini-rsllm** 是一个从零开始用 Rust 编写的最小化大语言模型（LLM）推理引擎。它能够加载真实的 GGUF 格式模型文件，并使用 LLaMA 架构生成文本。

这个项目的设计哲学是 **清晰优先于性能**——每一行代码都力求可读、可理解，但同时又是完全可运行的真实推理引擎。

### 它能做什么？

```bash
# 贪婪解码（确定性输出）
./target/release/mini-rsllm model.gguf -p "The capital of France is" -n 30 -t 0.0

# 创造性生成
./target/release/mini-rsllm model.gguf -p "Once upon a time" -n 100 -t 0.8
```

### 它包含什么？

| 模块 | 文件 | 功能 |
|------|------|------|
| GGUF 解析器 | `src/gguf.rs` | 解析模型文件格式 |
| 分词器 | `src/tokenizer.rs` | 文本 ↔ Token ID 转换 |
| 张量运算 | `src/tensor.rs` | 量化/反量化、矩阵乘法、核心算子 |
| LLaMA 模型 | `src/model.rs` | Transformer 前向传播 |
| 采样器 | `src/sampler.rs` | 温度、top-k、top-p 采样 |
| 后端抽象 | `src/backend.rs` | CPU/Metal GPU 加速 |
| CLI 入口 | `src/main.rs` | 参数解析、推理循环 |

### 为什么学习这个项目？

1. **完整性**：覆盖 LLM 推理的每一个环节——从文件解析到文本生成
2. **真实性**：能运行真实模型（TinyLlama 1.1B），不是玩具实现
3. **极简依赖**：仅 3 个外部 crate（`memmap2`、`byteorder`、`half`），核心逻辑全部手写
4. **Rust 实践**：展示 Rust 在系统编程中的优势——内存安全、零成本抽象、trait 多态

---

## 0.2 前置知识

### 必须掌握

- **Rust 基础**：所有权、借用、生命周期、trait、泛型、错误处理
- **基本数据结构**：`Vec`、`HashMap`、切片（slice）

### 建议了解

- **线性代数**：向量、矩阵乘法、点积
- **机器学习基础**：神经网络、激活函数、softmax
- **Transformer 概念**：注意力机制、编码器/解码器（不了解也没关系，第 4 章会详细讲解）

### 不需要

- GPU 编程经验（Metal 部分是可选的）
- 深度学习框架经验（PyTorch/TensorFlow）

---

## 0.3 环境搭建

### 第一步：安装 Rust 工具链

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

### 第二步：克隆并编译项目

```bash
git clone <repo-url> mini-rsllm
cd mini-rsllm
cargo build --release
```

### 第三步：下载测试模型

我们使用 TinyLlama-1.1B-Chat 的 Q4_0 量化版本（约 600MB）：

```bash
mkdir -p models
# 方式一：使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_0.gguf --local-dir models/

# 方式二：直接下载
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

### 第四步：运行第一个示例

```bash
./target/release/mini-rsllm models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -p "The meaning of life is" -n 50 -t 0.0
```

如果看到模型逐字输出文本，恭喜你，环境搭建成功！

---

## 0.4 项目架构速览

一次完整的推理过程如下：

```
用户输入 "Hello world"
        │
        ▼
┌─────────────────┐
│  1. GGUF Load   │  加载模型文件（内存映射，不拷贝到 RAM）
│     gguf.rs     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Tokenizer   │  "Hello world" → [1, 15043, 3186]
│  tokenizer.rs   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Model       │  对每个 token 执行前向传播
│   model.rs      │  经过 N 层 Transformer → 输出 logits
│   tensor.rs     │  （量化权重实时反量化）
│   backend.rs    │  （CPU 或 Metal GPU 加速）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Sampler     │  从 logits 中采样下一个 token
│   sampler.rs    │  温度 → top-k → softmax → top-p → 采样
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Decode      │  token ID → 文本片段 → 流式输出
│  tokenizer.rs   │
└────────┬────────┘
         │
         ▼
    重复 3-5 直到生成结束
```

### 模块依赖关系

```
main.rs ──→ gguf.rs        （加载模型文件）
   │──→ tokenizer.rs       （构建分词器）
   │──→ backend.rs         （初始化计算后端）
   │──→ model.rs           （创建模型）
   │──→ sampler.rs         （创建采样器）
   │
   └──→ 推理循环: model.forward() → sampler.sample() → tokenizer.decode()

model.rs ──→ backend.rs    （调用计算后端）
         ──→ tensor.rs     （张量运算）
         ──→ gguf.rs       （读取权重数据）

backend.rs ──→ tensor.rs   （CPU 后端直接调用）
           ──→ metal/      （Metal GPU 后端，可选）
```

---

## 0.5 学习路线图

本教程按照 **推理流程的数据流向** 组织章节：

| 章节 | 主题 | 对应文件 | 核心问题 |
|------|------|----------|----------|
| 第 1 章 | GGUF 文件格式 | `gguf.rs` | 模型文件里存了什么？怎么读？ |
| 第 2 章 | 分词器 | `tokenizer.rs` | 文本怎么变成数字？ |
| 第 3 章 | 张量与量化 | `tensor.rs` | 权重怎么压缩？数学运算怎么做？ |
| 第 4 章 | Transformer 基础 | （理论章节） | 注意力机制是什么？ |
| 第 5 章 | LLaMA 模型 | `model.rs` | 模型怎么从 token 生成 logits？ |
| 第 6 章 | 采样策略 | `sampler.rs` | 怎么从概率分布中选下一个词？ |
| 第 7 章 | 后端抽象 | `backend.rs` | CPU 和 GPU 怎么切换？ |
| 第 8 章 | 完整推理流程 | `main.rs` | 所有模块怎么串起来？ |
| 第 9 章 | 实践与扩展 | （动手项目） | 怎么修改和扩展这个引擎？ |

**建议学习方式**：

1. **通读**：先快速浏览每章的概述，建立全局认知
2. **精读**：逐章深入，对照源码理解每一行
3. **动手**：完成每章的练习题，修改代码观察效果
4. **回顾**：学完后重新阅读 `main.rs`，此时你应该能理解每一行

---

## 0.6 小结

✅ 你现在应该：

- [ ] 理解 mini-rsllm 的定位：一个教学用的最小化 LLM 推理引擎
- [ ] 成功编译并运行了项目
- [ ] 了解项目的模块结构和数据流向
- [ ] 明确接下来的学习路线

**下一章**：[第 1 章：GGUF 文件格式](./01-gguf-format.md) —— 我们从模型文件的加载开始，理解推理引擎的第一步。