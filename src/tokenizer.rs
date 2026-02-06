// BPE tokenizer: encode text→tokens, decode tokens→text
// Loads vocabulary and merge scores from GGUF metadata (SentencePiece-style BPE).

use crate::gguf::GgufFile;
use std::collections::HashMap;

pub struct Tokenizer {
    /// Token ID → token string
    vocab: Vec<String>,
    /// Token string → token ID (for encoding)
    token_to_id: HashMap<String, u32>,
    /// Merge priority scores (higher = merge first). Used when merges are absent.
    scores: Vec<f32>,
    /// BPE merge ranks: (token_a, token_b) -> (rank, merged_token_id)
    merge_ranks: Option<HashMap<(u32, u32), (usize, u32)>>,
    /// Token types (1=normal, 2=unknown, 3=control, 6=byte, etc.)
    token_types: Vec<i32>,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl Tokenizer {
    /// Load tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        // Read vocabulary tokens
        let tokens_val = gguf
            .get_meta("tokenizer.ggml.tokens")
            .expect("missing tokenizer.ggml.tokens");
        let tokens_arr = tokens_val.as_array().expect("tokens should be array");
        let vocab: Vec<String> = tokens_arr
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        // Build reverse mapping
        let mut token_to_id = HashMap::new();
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        // Read scores
        let scores = if let Some(scores_val) = gguf.get_meta("tokenizer.ggml.scores") {
            scores_val
                .as_array()
                .expect("scores should be array")
                .iter()
                .map(|v| v.as_f32().unwrap_or(0.0))
                .collect()
        } else {
            vec![0.0; vocab.len()]
        };

        // Read merges (BPE ranks)
        let merge_ranks = if let Some(merges_val) = gguf.get_meta("tokenizer.ggml.merges") {
            let merges = merges_val.as_array().expect("merges should be array");
            let mut ranks: HashMap<(u32, u32), (usize, u32)> = HashMap::new();
            for (rank, m) in merges.iter().enumerate() {
                let s = m.as_str().unwrap_or("");
                let mut it = s.split(' ');
                let a = it.next();
                let b = it.next();
                if a.is_none() || b.is_none() {
                    continue;
                }
                let a = a.unwrap();
                let b = b.unwrap();
                let id_a = match token_to_id.get(a) {
                    Some(v) => *v,
                    None => continue,
                };
                let id_b = match token_to_id.get(b) {
                    Some(v) => *v,
                    None => continue,
                };
                let merged = format!("{}{}", a, b);
                let id_merged = match token_to_id.get(&merged) {
                    Some(v) => *v,
                    None => continue,
                };
                ranks.insert((id_a, id_b), (rank, id_merged));
            }
            if ranks.is_empty() {
                None
            } else {
                Some(ranks)
            }
        } else {
            None
        };

        // Read token types
        let token_types = if let Some(types_val) = gguf.get_meta("tokenizer.ggml.token_type") {
            types_val
                .as_array()
                .expect("token_type should be array")
                .iter()
                .map(|v| v.as_i32().unwrap_or(1))
                .collect()
        } else {
            vec![1i32; vocab.len()]
        };

        // Read special token IDs
        let bos_id = gguf
            .get_meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);
        let eos_id = gguf
            .get_meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        Tokenizer {
            vocab,
            token_to_id,
            scores,
            merge_ranks,
            token_types,
            bos_id,
            eos_id,
        }
    }

    /// Encode text into token IDs using SentencePiece-style BPE.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::new();

        if add_bos {
            tokens.push(self.bos_id);
        }

        if text.is_empty() {
            return tokens;
        }

        // Start with individual UTF-8 bytes as byte-fallback tokens or
        // individual characters matched against the vocabulary.
        // First, try to initialize with per-character tokens (SentencePiece style).
        let mut work: Vec<u32> = Vec::new();

        // SentencePiece prepends a space (▁) before the text
        let text_with_space = format!("▁{}", text.replace(' ', "▁"));

        // Try single-character (or multi-byte char) lookup first
        for ch in text_with_space.chars() {
            let s = ch.to_string();
            if let Some(&id) = self.token_to_id.get(&s) {
                work.push(id);
            } else {
                // Fall back to byte tokens: <0xHH>
                for byte in s.as_bytes() {
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        work.push(id);
                    }
                    // If even byte fallback fails, skip (shouldn't happen with complete vocab)
                }
            }
        }

        // BPE merge loop: use explicit merges if present, otherwise fallback to scores
        if let Some(ranks) = &self.merge_ranks {
            self.bpe_merge_with_ranks(&mut work, ranks);
        } else {
            self.bpe_merge_with_scores(&mut work);
        }

        tokens.extend_from_slice(&work);
        tokens
    }

    /// Decode a single token ID to a string fragment.
    pub fn decode_token(&self, token_id: u32) -> String {
        let id = token_id as usize;
        if id >= self.vocab.len() {
            return String::new();
        }

        let token_str = &self.vocab[id];

        // Check if it's a byte-fallback token like <0xHH>
        if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
            if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                return String::from(byte as char);
            }
        }

        // Replace SentencePiece space marker with actual space
        token_str.replace('▁', " ")
    }

    /// Decode a sequence of token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        for &id in tokens {
            text.push_str(&self.decode_token(id));
        }
        text
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn bpe_merge_with_ranks(&self, work: &mut Vec<u32>, ranks: &HashMap<(u32, u32), (usize, u32)>) {
        loop {
            if work.len() < 2 {
                break;
            }
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

            if best_idx == usize::MAX {
                break;
            }

            work[best_idx] = best_id;
            work.remove(best_idx + 1);
        }
    }

    fn bpe_merge_with_scores(&self, work: &mut Vec<u32>) {
        loop {
            if work.len() < 2 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;

            for i in 0..work.len() - 1 {
                let merged = format!(
                    "{}{}",
                    self.vocab[work[i] as usize],
                    self.vocab[work[i + 1] as usize]
                );
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self.scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = id;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            work[best_idx] = best_id;
            work.remove(best_idx + 1);
        }
    }
}
