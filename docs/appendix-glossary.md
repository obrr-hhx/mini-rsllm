# 附录：术语表

> 本术语表收录了教程中出现的所有专业术语，按字母顺序排列。每个术语附有简短解释和首次出现的章节。

---

## A

**Attention（注意力机制）**：Transformer 的核心组件，让模型在处理每个 token 时能"关注"序列中的其他位置。通过 Query、Key、Value 三个矩阵计算注意力权重。→ 第 4 章

**Autoregressive（自回归）**：逐个生成 token 的方式——每次用已生成的所有 token 预测下一个。LLM 的生成过程就是自回归的。→ 第 8 章

## B

**Backend（后端）**：执行数学运算的抽象层。本项目支持 CPU 和 Metal（Apple GPU）两种后端。→ 第 7 章

**BOS（Beginning of Sequence）**：序列开始标记，通常是 token ID 1。编码时自动添加在序列最前面。→ 第 2 章

**BPE（Byte Pair Encoding，字节对编码）**：一种子词分词算法，通过反复合并最频繁的字符对来构建词汇表。→ 第 2 章

## C

**Causal Mask（因果掩码）**：注意力机制中的三角形掩码，确保每个位置只能看到它之前的 token，不能"偷看"未来。→ 第 4 章

**Context Length（上下文长度）**：模型能处理的最大 token 序列长度。超过此长度的输入会被截断。→ 第 5 章

## D

**Dequantize（反量化）**：将压缩的量化数据还原为 f32 浮点数的过程。在矩阵乘法时逐行进行。→ 第 3 章

**Dynamic Dispatch（动态分发）**：通过 `dyn Trait` 在运行时决定调用哪个实现。本项目用 `Box<dyn Backend>` 实现后端切换。→ 第 7 章

## E

**Embedding（嵌入）**：将离散的 token ID 映射为连续的向量表示。本质上是一个查表操作。→ 第 5 章

**EOS（End of Sequence）**：序列结束标记，通常是 token ID 2。模型生成此 token 时停止生成。→ 第 2 章

## F

**FFI（Foreign Function Interface）**：跨语言调用接口。本项目通过 FFI 从 Rust 调用 Objective-C 编写的 Metal 代码。→ 第 7 章

**FFN（Feed-Forward Network，前馈网络）**：Transformer 中的全连接层，对每个位置独立进行非线性变换。本项目使用 SwiGLU 变体。→ 第 4 章

**Forward Pass（前向传播）**：将输入数据通过模型的所有层，计算输出的过程。一次 forward 处理一个 token。→ 第 5 章

## G

**GGUF（GPT-Generated Unified Format）**：一种二进制模型文件格式，包含元数据、张量信息和量化权重。由 llama.cpp 项目定义。→ 第 1 章

**GQA（Grouped-Query Attention，分组查询注意力）**：多个 Q 头共享一组 KV 头的注意力变体。减少 KV 缓存大小，如 32:4 的比例。→ 第 4 章

**Greedy Decoding（贪婪解码）**：每步选择概率最高的 token。等价于 temperature = 0。确定性但可能单调。→ 第 6 章

## H

**Head Dimension（头维度）**：每个注意力头的向量维度。通常 head_dim = dim / n_heads。→ 第 4 章

## K

**KV Cache（键值缓存）**：缓存已计算的 Key 和 Value 向量，避免在自回归生成时重复计算。→ 第 4 章、第 5 章

## L

**Layer Offloading（层级卸载）**：将部分 Transformer 层的计算从 CPU 转移到 GPU。本项目从最后一层开始卸载。→ 第 7 章

**Logits**：模型输出的原始分数（未归一化），维度等于词汇表大小。经过 softmax 后变为概率分布。→ 第 5 章、第 6 章

## M

**Matmul（矩阵乘法）**：最核心的计算操作。在推理中表现为矩阵-向量乘法（matmul_vec），占 90%+ 的计算时间。→ 第 3 章

**Metal**：Apple 的 GPU 编程框架，用于在 Apple Silicon 上加速计算。→ 第 7 章

**mmap（Memory Mapping，内存映射）**：将文件映射到虚拟内存地址空间，由操作系统按需加载页面。避免将整个模型文件读入 RAM。→ 第 1 章

## N

**Nucleus Sampling**：见 Top-p Sampling。

## P

**Prefill（预填充）**：处理 prompt 中所有 token 以填充 KV 缓存的阶段。只有最后一个 token 的 logits 用于采样。→ 第 8 章

**PRNG（Pseudo-Random Number Generator）**：伪随机数生成器。本项目使用 Xorshift64 算法。→ 第 6 章

## Q

**Q4_0 / Q8_0 / Q6_K**：不同的量化格式。数字表示每个元素的位数，后缀表示变体。Q4_0 = 4 bit 均匀量化。→ 第 3 章

**Quantization（量化）**：将 f32 权重压缩为低精度格式（如 4-bit）以减少模型大小和内存使用。→ 第 3 章

## R

**Residual Connection（残差连接）**：将层的输入直接加到输出上（output = input + layer(input)），帮助梯度流动和训练稳定性。→ 第 4 章

**RMS Norm（Root Mean Square Normalization）**：一种归一化方法，比 Layer Norm 更简单（无偏置项）。公式：x * weight / sqrt(mean(x²) + eps)。→ 第 3 章

**RoPE（Rotary Position Embedding，旋转位置编码）**：通过旋转向量来编码位置信息。频率随维度指数衰减，低维编码局部位置，高维编码全局位置。→ 第 3 章、第 4 章

## S

**Sampler（采样器）**：从 logits 中选择下一个 token 的组件。包含温度缩放、top-k、top-p 等策略。→ 第 6 章

**SentencePiece**：一种分词方式，使用 `▁`（U+2581）标记单词边界（空格位置）。→ 第 2 章

**SiLU（Sigmoid Linear Unit）**：激活函数，公式：silu(x) = x * sigmoid(x)。用于 SwiGLU FFN。→ 第 3 章

**Softmax**：将任意实数向量转换为概率分布的函数。公式：softmax(x_i) = exp(x_i) / Σexp(x_j)。→ 第 3 章

**SwiGLU**：一种门控 FFN 变体，使用 SiLU 激活和门控机制。公式：SwiGLU(x) = (W_gate · x ⊙ silu(W_up · x)) · W_down。→ 第 4 章

## T

**Temperature（温度）**：控制采样随机性的参数。logits /= temperature。温度越高越随机，0 = 贪婪解码。→ 第 6 章

**Tensor（张量）**：多维数组，存储模型权重和中间计算结果。→ 第 3 章

**Token**：文本的最小处理单元。可以是一个字、子词或字符。每个 token 有唯一的整数 ID。→ 第 2 章

**Top-k Sampling**：只保留概率最高的 k 个 token，其余设为 0。k=1 等价于贪婪解码。→ 第 6 章

**Top-p Sampling（Nucleus Sampling）**：按概率从高到低累加，保留累积概率达到 p 的最小 token 集合。自适应地调整候选数量。→ 第 6 章

**Transformer**：一种基于注意力机制的神经网络架构，是现代 LLM 的基础。由 Vaswani et al. 2017 提出。→ 第 4 章

## X

**Xorshift64**：一种简单高效的伪随机数生成算法，使用三次异或移位操作。本项目用于采样时的随机数生成。→ 第 6 章

---

> **提示**：遇到不熟悉的术语时，可以回到对应章节查看详细解释和代码实现。
