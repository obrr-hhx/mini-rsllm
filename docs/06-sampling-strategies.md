# 第 6 章：采样策略

> **学习目标**：理解从 logits 到 token 的采样过程，掌握温度、top-k、top-p 的原理和实现。
>
> **对应源码**：`src/sampler.rs`（226 行）
>
> **预计时间**：3-4 小时

---

## 6.1 为什么需要采样？

模型的 `forward()` 输出的是 logits——每个词的"原始得分"。我们需要从中选择下一个 token。

最简单的方法是选得分最高的（贪心），但这会导致生成文本重复、无趣。采样策略通过引入可控的随机性，让生成更加多样和自然。

```
logits = [1.2, -0.5, 3.4, 0.8, ...]  (vocab_size 个值)
                    │
            ┌───────┼───────┐
            ▼       ▼       ▼
         温度缩放  top-k   top-p
            │       │       │
            └───────┼───────┘
                    ▼
              softmax → 概率分布
                    │
                    ▼
              随机采样 → token ID
```

---

## 6.2 采样流水线

### 配置结构

```rust
pub struct SamplerConfig {
    pub temperature: f32,  // 温度：0=贪心，>1=更随机，<1=更确定
    pub top_k: usize,      // 只保留概率最高的 k 个候选
    pub top_p: f32,        // 只保留累积概率达到 p 的候选
    pub seed: u64,         // 随机种子（0 使用默认值 42）
}
```

### 完整采样流程

```rust
pub fn sample(&mut self, logits: &mut Vec<f32>) -> u32 {
    // 1. 温度为 0 → 贪心（直接选最大值）
    if self.config.temperature <= 0.0 {
        return argmax(logits);
    }

    // 2. 温度缩放
    for v in logits.iter_mut() {
        *v /= self.config.temperature;
    }

    // 3. Top-k 过滤
    if self.config.top_k > 0 && self.config.top_k < logits.len() {
        self.apply_top_k(logits);
    }

    // 4. Softmax → 概率分布
    tensor::softmax(logits);

    // 5. Top-p 过滤
    if self.config.top_p < 1.0 {
        self.apply_top_p(logits);
    }

    // 6. 加权随机采样
    self.sample_from_probs(logits)
}
```

---

## 6.3 温度（Temperature）

温度控制概率分布的"尖锐程度"：

```
logits = [3.0, 1.0, 0.5]

温度 = 0.5 (低温，更确定):
  logits/T = [6.0, 2.0, 1.0]
  softmax  = [0.94, 0.02, 0.01]  ← 几乎确定选第一个

温度 = 1.0 (标准):
  logits/T = [3.0, 1.0, 0.5]
  softmax  = [0.84, 0.11, 0.07]

温度 = 2.0 (高温，更随机):
  logits/T = [1.5, 0.5, 0.25]
  softmax  = [0.51, 0.19, 0.15]  ← 分布更均匀
```

**直觉**：温度越低，模型越"自信"（倾向选概率最高的）；温度越高，模型越"随意"（各选项概率更接近）。

---

## 6.4 Top-k 过滤

只保留概率最高的 k 个候选，其余设为 -∞：

```rust
fn apply_top_k(&self, logits: &mut Vec<f32>) {
    let k = self.config.top_k;

    // 按值降序排列，找到第 k 大的值
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let threshold = indexed[k - 1].1;
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;  // 排除在外
        }
    }
}
```

**示例**（top_k=3）：

```
logits = [1.2, 3.4, 0.5, 2.8, 0.1, 1.9]
排序后前 3: [3.4, 2.8, 1.9]
threshold = 1.9

过滤后: [-∞, 3.4, -∞, 2.8, -∞, 1.9]
softmax: [0.0, 0.55, 0.0, 0.30, 0.0, 0.15]
```

---

## 6.5 Top-p（核采样）

Top-p 更灵活——保留累积概率达到 p 的最少候选：

```rust
fn apply_top_p(&self, probs: &mut Vec<f32>) {
    // 按概率降序排列
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 累加概率直到达到 top_p
    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= self.config.top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // 截断后的候选概率置零
    for &(idx, _) in indexed[cutoff_idx..].iter() {
        probs[idx] = 0.0;
    }

    // 重新归一化
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }
}
```

**示例**（top_p=0.9）：

```
概率（降序）: [0.50, 0.25, 0.15, 0.05, 0.03, 0.02]
累积:         [0.50, 0.75, 0.90, ...]
                                  ↑ 达到 0.9，截断

保留前 3 个，重新归一化:
[0.556, 0.278, 0.167, 0.0, 0.0, 0.0]
```

**Top-k vs Top-p**：
- Top-k 固定候选数量，不管概率分布如何
- Top-p 自适应——分布集中时候选少，分布分散时候选多

---

## 6.6 加权随机采样

经过温度、top-k、top-p 处理后，我们得到了一个概率分布。最后一步是从中随机选择一个 token：

```rust
fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
    let r = self.random_f32();       // 生成 [0, 1) 的随机数
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;         // 落在这个区间，选择此 token
        }
    }
    (probs.len() - 1) as u32         // 兜底：返回最后一个
}
```

**原理**：将 [0, 1) 区间按概率分割，随机数落在哪个区间就选哪个 token：

```
概率: [0.55, 0.30, 0.15]

[0, 1) 区间分割:
|←── 0.55 ──→|←─ 0.30 ─→|←0.15→|
0            0.55        0.85    1.0

r = 0.42 → 落在第一段 → 选 token 0
r = 0.71 → 落在第二段 → 选 token 1
r = 0.93 → 落在第三段 → 选 token 2
```

概率越大的 token，占据的区间越宽，被选中的概率越高。

---

## 6.7 Xorshift64 伪随机数生成器

采样需要随机数，但我们不使用标准库的随机数生成器，而是自己实现了一个极简的 **Xorshift64**：

```rust
fn random_f32(&mut self) -> f32 {
    self.rng_state ^= self.rng_state << 13;
    self.rng_state ^= self.rng_state >> 7;
    self.rng_state ^= self.rng_state << 17;
    (self.rng_state as f32) / (u64::MAX as f32)
}
```

### 为什么不用标准库？

1. **可复现性**：相同种子必须产生完全相同的序列，跨平台、跨版本一致
2. **零依赖**：不引入额外的 crate
3. **简单透明**：3 行位运算，容易理解和验证
4. **足够好**：对于文本生成，不需要密码学级别的随机性

### 种子初始化

```rust
pub fn new(config: SamplerConfig) -> Self {
    let rng_state = if config.seed == 0 { 42 } else { config.seed };
    Sampler { config, rng_state }
}
```

种子为 0 时使用默认值 42，确保状态不为零（Xorshift 的零状态会永远输出零）。

### 可复现性验证

项目的测试验证了相同种子产生相同结果：

```rust
#[test]
fn fixed_seed_sampling_is_reproducible() {
    let mut a = Sampler::new(SamplerConfig { seed: 42, ... });
    let mut b = Sampler::new(SamplerConfig { seed: 42, ... });

    for step in 0..200 {
        let mut logits_a = make_logits(step, 128);
        let mut logits_b = logits_a.clone();
        assert_eq!(a.sample(&mut logits_a), b.sample(&mut logits_b));
    }
}
```

---

## 6.8 贪心采样（Argmax）

当温度为 0 时，直接选择得分最高的 token：

```rust
fn argmax(values: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}
```

贪心采样完全确定——相同输入永远产生相同输出。适合需要一致性的场景（如测试、评估），但生成文本往往重复乏味。

---

## 6.9 小结

✅ 本章你学到了：

- [ ] 采样流水线：温度 → top-k → softmax → top-p → 随机采样
- [ ] 温度：控制概率分布的尖锐程度
- [ ] Top-k：固定数量的候选过滤
- [ ] Top-p：自适应的累积概率过滤
- [ ] 加权随机采样：累积概率区间法
- [ ] Xorshift64：极简可复现的伪随机数生成器
- [ ] 贪心采样：温度为 0 时的确定性选择

### 关键设计决策

1. **为什么 top-k 在 softmax 之前，top-p 在之后？** Top-k 操作的是 logits（设为 -∞），top-p 操作的是概率（需要归一化）。顺序不能颠倒。
2. **为什么用自定义 PRNG？** 标准库的随机数生成器可能跨版本不一致，自定义 Xorshift64 保证完全可复现。
3. **为什么 top-p 后要重新归一化？** 置零后概率之和不再为 1，必须重新归一化才能正确采样。

**下一章**：[第 7 章：后端抽象与 GPU 加速](./07-backend-abstraction.md) —— 理解如何通过 trait 实现 CPU/GPU 的统一接口。
