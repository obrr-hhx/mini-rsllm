# 第 2 章：分词器原理与实现

> **学习目标**：理解 BPE 分词算法，掌握 SentencePiece 风格的编码/解码流程，能够调试分词问题。
>
> **对应源码**：`src/tokenizer.rs`（268 行）
>
> **预计时间**：3-4 小时

---

## 2.1 为什么需要分词？

语言模型不能直接处理文本字符串——它需要数字。分词器（Tokenizer）的作用就是在文本和数字之间建立桥梁：

```
编码（Encode）：  "Hello world"  →  [1, 15043, 3186]
解码（Decode）：  [1, 15043, 3186]  →  "Hello world"
```

### 分词粒度的选择

| 粒度 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| 字符级 | `['H','e','l','l','o']` | 词表小（~256） | 序列太长，语义弱 |
| 词级 | `['Hello', 'world']` | 语义清晰 | 词表巨大，无法处理新词 |
| **子词级** | `['Hello', '▁world']` | **平衡**：词表适中，能处理新词 | 需要学习合并规则 |

现代 LLM 几乎都使用 **子词级分词**，其中最流行的算法是 **BPE（Byte Pair Encoding）**。

---

## 2.2 BPE 算法原理

BPE 的核心思想非常简单：**反复合并最频繁出现的相邻字符对**。

### 训练阶段（离线，不在本项目中）

```
初始词表：所有单个字符 {'a', 'b', 'c', ..., '▁', ...}

语料统计：
  "▁the" 出现 1000 次
  "▁is"  出现 800 次
  ...

第 1 轮合并：最频繁的对是 ('t', 'h') → 合并为 'th'
第 2 轮合并：最频繁的对是 ('th', 'e') → 合并为 'the'
第 3 轮合并：最频繁的对是 ('▁', 'the') → 合并为 '▁the'
...
重复直到词表达到目标大小（如 32000）
```

### 推理阶段（本项目实现的部分）

编码时，我们需要 **重放合并过程**：

```
输入: "▁Hello"
初始: ['▁', 'H', 'e', 'l', 'l', 'o']

查找可合并的对，选择优先级最高的：
  ('l', 'l') → 'll'  (rank=42)
  ('H', 'e') → 'He'  (rank=100)
  ...

合并 rank 最小的 → ['▁', 'H', 'e', 'll', 'o']
继续合并 → ['▁', 'He', 'll', 'o']
继续合并 → ['▁', 'Hell', 'o']
继续合并 → ['▁', 'Hello']
继续合并 → ['▁Hello']
无法继续合并 → 结束

结果: ['▁Hello'] → token ID: [15043]
```

---

## 2.3 代码解读：分词器加载

### 数据结构

```rust
pub struct Tokenizer {
    vocab: Vec<String>,                                    // ID → 字符串
    token_to_id: HashMap<String, u32>,                     // 字符串 → ID
    scores: Vec<f32>,                                      // 合并优先级分数
    merge_ranks: Option<HashMap<(u32, u32), (usize, u32)>>,// BPE 合并规则
    token_types: Vec<i32>,                                 // token 类型标记
    pub bos_id: u32,                                       // 序列开始 token
    pub eos_id: u32,                                       // 序列结束 token
}
```

分词器有两种合并策略：
- **merge_ranks**：显式的合并规则表 `(token_a, token_b) → (rank, merged_id)`
- **scores**：隐式的优先级分数，合并后的 token 分数越高越优先

### 从 GGUF 加载

```rust
pub fn from_gguf(gguf: &GgufFile) -> Self {
    // 1. 读取词表：tokenizer.ggml.tokens → ["<unk>", "<s>", "</s>", "▁", "t", ...]
    let vocab: Vec<String> = tokens_arr.iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();

    // 2. 构建反向映射：{"<unk>": 0, "<s>": 1, "</s>": 2, "▁": 3, ...}
    let mut token_to_id = HashMap::new();
    for (i, tok) in vocab.iter().enumerate() {
        token_to_id.insert(tok.clone(), i as u32);
    }

    // 3. 读取合并分数：tokenizer.ggml.scores
    // 4. 读取合并规则：tokenizer.ggml.merges（如果存在）
    // 5. 读取特殊 token ID：bos_id, eos_id
}
```

### 合并规则的解析

```rust
// merges 格式："▁ t" 表示 token "▁" 和 "t" 可以合并为 "▁t"
let merge_ranks = if let Some(merges_val) = gguf.get_meta("tokenizer.ggml.merges") {
    let mut ranks: HashMap<(u32, u32), (usize, u32)> = HashMap::new();
    for (rank, m) in merges.iter().enumerate() {
        let s = m.as_str().unwrap_or("");
        let mut it = s.split(' ');
        let a = it.next().unwrap();  // "▁"
        let b = it.next().unwrap();  // "t"
        let id_a = token_to_id[a];   // 查找 "▁" 的 ID
        let id_b = token_to_id[b];   // 查找 "t" 的 ID
        let merged = format!("{}{}", a, b);  // "▁t"
        let id_merged = token_to_id[&merged]; // 查找 "▁t" 的 ID
        ranks.insert((id_a, id_b), (rank, id_merged));
    }
    Some(ranks)
};
```

> **关键理解**：`rank` 越小，优先级越高。rank=0 的合并最先执行。

---

## 2.4 代码解读：编码流程

```rust
pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
    let mut tokens: Vec<u32> = Vec::new();

    // 第一步：添加 BOS（Begin of Sequence）token
    if add_bos {
        tokens.push(self.bos_id);  // 通常是 token ID 1
    }

    // 第二步：SentencePiece 预处理——在文本前加 ▁，空格替换为 ▁
    let text_with_space = format!("▁{}", text.replace(' ', "▁"));
    // "Hello world" → "▁Hello▁world"
```



### 字符级初始化

```rust
    // SentencePiece 预处理：在文本前加 ▁，空格替换为 ▁
    let text_with_space = format!("▁{}", text.replace(' ', "▁"));
    // "Hello world" → "▁Hello▁world"

    let mut work: Vec<u32> = Vec::new();

    // 逐字符查找词表
    for ch in text_with_space.chars() {
        let s = ch.to_string();
        if let Some(&id) = self.token_to_id.get(&s) {
            work.push(id);  // 找到了，直接用
        } else {
            // 字节回退：将字符拆成 UTF-8 字节
            for byte in s.as_bytes() {
                let byte_token = format!("<0x{:02X}>", byte);
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    work.push(id);
                }
            }
        }
    }
```

**字节回退机制**：当某个字符不在词表中时（比如罕见的 Unicode 字符），将其拆成 UTF-8 字节，用 `<0x??>`格式的特殊 token 表示。例如中文字符 "你" 的 UTF-8 编码是 `E4 BD A0`，会被编码为 `[<0xE4>, <0xBD>, <0xA0>]`。

### BPE 合并过程

初始化后，`work` 数组包含字符级的 token ID。接下来执行 BPE 合并：

```rust
    // 选择合并策略
    if let Some(ranks) = &self.merge_ranks {
        self.bpe_merge_with_ranks(&mut work, ranks);  // 使用显式合并规则
    } else {
        self.bpe_merge_with_scores(&mut work);         // 使用分数回退
    }
```

#### 策略一：基于 Rank 的合并（优先）

```rust
fn bpe_merge_with_ranks(&self, work: &mut Vec<u32>,
                         ranks: &HashMap<(u32, u32), (usize, u32)>) {
    loop {
        if work.len() < 2 { break; }

        // 扫描所有相邻对，找 rank 最小的
        let mut best_rank = usize::MAX;
        let mut best_idx = usize::MAX;
        let mut best_id = 0u32;

        for i in 0..work.len() - 1 {
            if let Some((rank, merged_id)) = ranks.get(&(work[i], work[i + 1])) {
                if *rank < best_rank {
                    best_rank = *rank;
                    best_idx = i;
                    best_id = *merged_id;
                }
            }
        }

        if best_idx == usize::MAX { break; }  // 没有可合并的对了

        // 执行合并：替换 work[best_idx]，删除 work[best_idx+1]
        work[best_idx] = best_id;
        work.remove(best_idx + 1);
    }
}
```

**执行过程示例**：

```
输入: "▁Hello"
初始: ['▁'(3), 'H'(72), 'e'(68), 'l'(75), 'l'(75), 'o'(78)]

第 1 轮: 扫描所有相邻对的 rank
  ('l','l') → rank=42, merged='ll'
  ('H','e') → rank=100, merged='He'
  ...
  选择 rank 最小的 ('l','l') → 合并
  结果: ['▁', 'H', 'e', 'll', 'o']

第 2 轮: ('e','ll') → rank=55 → 合并
  结果: ['▁', 'H', 'ell', 'o']

第 3 轮: ('H','ell') → rank=80 → 合并
  结果: ['▁', 'Hell', 'o']

第 4 轮: ('Hell','o') → rank=120 → 合并
  结果: ['▁', 'Hello']

第 5 轮: ('▁','Hello') → rank=5000 → 合并
  结果: ['▁Hello']

第 6 轮: 只剩 1 个 token，结束
最终: ['▁Hello'] → [15043]
```

#### 策略二：基于 Score 的合并（回退）

```rust
fn bpe_merge_with_scores(&self, work: &mut Vec<u32>) {
    loop {
        if work.len() < 2 { break; }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = usize::MAX;
        let mut best_id = 0u32;

        for i in 0..work.len() - 1 {
            // 拼接两个 token 的字符串，查找是否在词表中
            let merged = format!("{}{}", self.vocab[work[i] as usize],
                                         self.vocab[work[i + 1] as usize]);
            if let Some(&id) = self.token_to_id.get(&merged) {
                let score = self.scores[id as usize];
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                    best_id = id;
                }
            }
        }

        if best_idx == usize::MAX { break; }

        work[best_idx] = best_id;
        work.remove(best_idx + 1);
    }
}
```

**两种策略的区别**：
- **Rank 策略**：使用预计算的合并顺序表，rank 越小越优先。查找 O(1)。
- **Score 策略**：使用词表中的分数，分数越高越优先。需要拼接字符串查找，较慢。

---

## 2.5 代码解读：解码流程

解码是编码的逆过程——将 token ID 转回文本：

```rust
pub fn decode_token(&self, token_id: u32) -> String {
    let id = token_id as usize;
    if id >= self.vocab.len() {
        return String::new();
    }

    let token_str = &self.vocab[id];

    // 处理字节回退 token：<0xHH> → 对应的字节
    if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
            return String::from(byte as char);
        }
    }

    // 将 ▁ 替换回空格
    token_str.replace('▁', " ")
}
```

### 解码流程示例

```
输入 token IDs: [1, 15043, 3186]

token 1   → "<s>"     → "" (BOS，通常不输出)
token 15043 → "▁Hello" → " Hello" (▁ 替换为空格)
token 3186  → "▁world" → " world"

拼接: " Hello world"
去掉开头空格: "Hello world"
```

### 字节回退的解码

```
token "<0xE4>" → 0xE4 → 字节
token "<0xBD>" → 0xBD → 字节
token "<0xA0>" → 0xA0 → 字节

三个字节 [0xE4, 0xBD, 0xA0] → UTF-8 解码 → "你"
```

---

## 2.6 小结

✅ 本章你学到了：

- [ ] BPE 算法的核心思想：反复合并最频繁的相邻对
- [ ] SentencePiece 风格：用 ▁ 标记空格，支持无空格语言
- [ ] 编码流程：预处理 → 字符级初始化 → 字节回退 → BPE 合并
- [ ] 解码流程：查词表 → 处理字节回退 → 替换空格标记
- [ ] 两种合并策略：Rank（显式规则）vs Score（分数回退）

### 关键设计决策

1. **为什么用 BPE 而不是词级分词？** BPE 可以处理任意新词——最坏情况下拆成字节，永远不会遇到 "未知词"。
2. **为什么需要字节回退？** 确保任何 UTF-8 文本都能被编码，即使词表中没有对应的字符。
3. **为什么有两种合并策略？** 不同模型的 GGUF 文件可能只提供 merges 或只提供 scores，需要兼容两种情况。

**下一章**：[第 3 章：张量与量化](./03-tensor-and-quantization.md) —— 理解权重如何压缩存储，以及核心数学运算的实现。