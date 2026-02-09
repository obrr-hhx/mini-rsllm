# mini-rsllm 教学文档

> 从零理解 LLM 推理引擎——基于 Rust 的完整实现解析

---

**mini-rsllm** 是一个旨在学习本地LLM推理的教学型项目，代码总行数不超过 8000 行，核心 Rust 代码行数 ~4000 行。

## 📚 目录

| 章节 | 主题 | 核心问题 | 预计时间 |
|------|------|----------|----------|
| [第 0 章](./00-introduction.md) | **导读与环境搭建** | 这个项目是什么？怎么运行？ | 1-2h |
| [第 1 章](./01-gguf-format.md) | **GGUF 文件格式** | 模型文件里存了什么？怎么读？ | 3-4h |
| [第 2 章](./02-tokenizer.md) | **分词器** | 文本怎么变成数字？ | 3-4h |
| [第 3 章](./03-tensor-and-quantization.md) | **张量与量化** | 权重怎么压缩？数学运算怎么做？ | 4-5h |
| [第 4 章](./04-transformer-basics.md) | **Transformer 基础** | 注意力机制是什么？ | 3-4h |
| [第 5 章](./05-llama-model.md) | **LLaMA 模型实现** | 模型怎么从 token 生成 logits？ | 6-8h |
| [第 6 章](./06-sampling-strategies.md) | **采样策略** | 怎么从概率分布中选下一个词？ | 3-4h |
| [第 7 章](./07-backend-abstraction.md) | **后端抽象与 GPU 加速** | CPU 和 GPU 怎么切换？ | 4-5h |
| [第 8 章](./08-inference-pipeline.md) | **完整推理流程** | 所有模块怎么串起来？ | 3-4h |
| [第 9 章](./09-hands-on-projects.md) | **实践与扩展** | 怎么修改和扩展这个引擎？ | 10-15h |
| [附录](./appendix-glossary.md) | **术语表** | 快速查阅专业术语 | - |

**总预计学习时间**：40-60 小时

---

## 🗺️ 学习路径

```
数据流向：
GGUF 文件 → 分词器 → 张量运算 → Transformer → 模型 → 采样 → 后端 → 推理循环

对应章节：
  第1章  →  第2章  →  第3章  →  第4章  → 第5章 → 第6章 → 第7章 →  第8章
```

### 推荐学习方式

1. **通读**：先快速浏览每章的概述和流程图，建立全局认知
2. **精读**：逐章深入，对照 `src/` 下的源码理解每一行
3. **动手**：完成第 9 章的实践项目，修改代码观察效果
4. **回顾**：学完后重新阅读 `src/main.rs`，此时你应该能理解每一行

---

## 📁 源码对照

| 源文件 | 行数 | 对应章节 |
|--------|------|----------|
| `src/gguf.rs` | ~295 | 第 1 章 |
| `src/tokenizer.rs` | ~268 | 第 2 章 |
| `src/tensor.rs` | ~396 | 第 3 章 |
| `src/model.rs` | ~396 | 第 4、5 章 |
| `src/sampler.rs` | ~226 | 第 6 章 |
| `src/backend.rs` | ~552 | 第 7 章 |
| `src/metal/` | ~多文件 | 第 7 章 |
| `src/main.rs` | ~273 | 第 8 章 |

---

## ⚡ 快速开始

```bash
# 1. 编译
cargo build --release

# 2. 下载模型
mkdir -p models
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_0.gguf --local-dir models/

# 3. 运行
./target/release/mini-rsllm models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -p "The meaning of life is" -n 50 -t 0.0
```

---

## 🔧 前置知识

- **必须**：Rust 基础（所有权、trait、泛型）
- **建议**：线性代数基础（向量、矩阵乘法）
- **可选**：机器学习基础（神经网络、softmax）

不需要 GPU 编程经验或深度学习框架经验。
