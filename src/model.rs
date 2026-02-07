// LLaMA model: load config/weights from GGUF, forward pass with GQA attention.

use crate::backend::{Backend, CpuBackend};
use crate::gguf::GgufFile;
use crate::tensor::{dequantize, TensorRef};

/// LLaMA model configuration, read from GGUF metadata.
pub struct LlamaConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub ff_dim: usize,
    pub norm_eps: f32,
    pub rope_freq_base: f32,
    pub max_seq_len: usize,
    pub head_dim: usize,
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let dim = gguf
            .get_u32("llama.embedding_length")
            .unwrap_or(gguf.get_u32("gpt2.embedding_length").unwrap_or(512))
            as usize;
        let n_layers =
            gguf.get_u32("llama.block_count")
                .unwrap_or(gguf.get_u32("gpt2.block_count").unwrap_or(4)) as usize;
        let n_heads = gguf
            .get_u32("llama.attention.head_count")
            .unwrap_or(gguf.get_u32("gpt2.attention.head_count").unwrap_or(8))
            as usize;
        let n_kv_heads = gguf.get_u32("llama.attention.head_count_kv").unwrap_or(
            gguf.get_u32("gpt2.attention.head_count_kv")
                .unwrap_or(n_heads as u32),
        ) as usize;
        let ff_dim = gguf.get_u32("llama.feed_forward_length").unwrap_or(
            gguf.get_u32("gpt2.feed_forward_length")
                .unwrap_or((dim * 4) as u32),
        ) as usize;

        let vocab_size = gguf
            .get_meta("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(32000);

        // norm_eps can be f32 or f64
        let norm_eps = gguf
            .get_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or_else(|| {
                gguf.get_meta("llama.attention.layer_norm_rms_epsilon")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1e-5)
            });

        let rope_freq_base = gguf.get_f32("llama.rope.freq_base").unwrap_or(10000.0);

        let max_seq_len = gguf.get_u32("llama.context_length").unwrap_or(2048) as usize;

        let head_dim = dim / n_heads;

        LlamaConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            ff_dim,
            norm_eps,
            rope_freq_base,
            max_seq_len,
            head_dim,
        }
    }
}

/// Weight references for a single transformer layer.
pub struct LayerWeights<'a> {
    pub attn_norm: TensorRef<'a>,
    pub wq: TensorRef<'a>,
    pub wk: TensorRef<'a>,
    pub wv: TensorRef<'a>,
    pub wo: TensorRef<'a>,
    pub ffn_norm: TensorRef<'a>,
    pub w1: TensorRef<'a>, // gate
    pub w2: TensorRef<'a>, // down
    pub w3: TensorRef<'a>, // up
}

/// All model weights, referencing data in the GGUF mmap.
pub struct LlamaWeights<'a> {
    pub token_embd: TensorRef<'a>,
    pub output_norm: TensorRef<'a>,
    pub output: TensorRef<'a>,
    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> LlamaWeights<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, config: &LlamaConfig) -> Self {
        let token_embd = make_ref(gguf, "token_embd.weight");
        let output_norm = make_ref(gguf, "output_norm.weight");

        // Some models share token_embd as the output projection (tied weights)
        let output = if gguf.tensors.contains_key("output.weight") {
            make_ref(gguf, "output.weight")
        } else {
            make_ref(gguf, "token_embd.weight")
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(LayerWeights {
                attn_norm: make_ref(gguf, &format!("blk.{}.attn_norm.weight", i)),
                wq: make_ref(gguf, &format!("blk.{}.attn_q.weight", i)),
                wk: make_ref(gguf, &format!("blk.{}.attn_k.weight", i)),
                wv: make_ref(gguf, &format!("blk.{}.attn_v.weight", i)),
                wo: make_ref(gguf, &format!("blk.{}.attn_output.weight", i)),
                ffn_norm: make_ref(gguf, &format!("blk.{}.ffn_norm.weight", i)),
                w1: make_ref(gguf, &format!("blk.{}.ffn_gate.weight", i)),
                w2: make_ref(gguf, &format!("blk.{}.ffn_down.weight", i)),
                w3: make_ref(gguf, &format!("blk.{}.ffn_up.weight", i)),
            });
        }

        LlamaWeights {
            token_embd,
            output_norm,
            output,
            layers,
        }
    }
}

fn make_ref<'a>(gguf: &'a GgufFile, name: &str) -> TensorRef<'a> {
    let info = gguf
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor: {}", name))
        .clone();
    let data = gguf.tensor_data(&info);
    TensorRef { data, info }
}

/// KV cache for all layers.
pub struct KvCache {
    /// key_cache[layer][pos * kv_dim .. (pos+1) * kv_dim]
    pub key_cache: Vec<Vec<f32>>,
    /// value_cache[layer][pos * kv_dim .. (pos+1) * kv_dim]
    pub val_cache: Vec<Vec<f32>>,
    kv_dim: usize,
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

/// The full LLaMA model: config, weights, and KV cache.
pub struct LlamaModel<'a> {
    pub config: LlamaConfig,
    pub weights: LlamaWeights<'a>,
    pub kv_cache: KvCache,
    backend: Box<dyn Backend>,
    cpu_backend: CpuBackend,
}

impl<'a> LlamaModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile) -> Self {
        Self::from_gguf_with_backend(gguf, Box::new(CpuBackend::new()))
    }

    pub fn from_gguf_with_backend(gguf: &'a GgufFile, backend: Box<dyn Backend>) -> Self {
        let config = LlamaConfig::from_gguf(gguf);
        let weights = LlamaWeights::from_gguf(gguf, &config);
        let kv_dim = config.n_kv_heads * config.head_dim;
        let kv_cache = KvCache::new(config.n_layers, kv_dim, config.max_seq_len);

        LlamaModel {
            config,
            weights,
            kv_cache,
            backend,
            cpu_backend: CpuBackend::new(),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Forward pass for a single token at a given position.
    /// Returns logits vector of shape [vocab_size].
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let w = &self.weights;
        let head_dim = cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * head_dim;
        let n_heads_per_kv = cfg.n_heads / cfg.n_kv_heads;

        if pos >= cfg.max_seq_len {
            return Err(format!(
                "position {} exceeds context length {}",
                pos, cfg.max_seq_len
            ));
        }
        if token_id as usize >= w.token_embd.rows() {
            return Err(format!(
                "token id {} out of embedding range {}",
                token_id,
                w.token_embd.rows()
            ));
        }

        // 1. Embedding lookup: x = token_embd[token_id] → [dim]
        let embd_row = w.token_embd.dequantize_row(token_id as usize);
        let mut x = embd_row;

        // 2. Transformer layers
        for layer_idx in 0..cfg.n_layers {
            let layer = &w.layers[layer_idx];
            let use_gpu = self.backend.should_use_layer_gpu(layer_idx, cfg.n_layers);

            // Dequantize norm weights (1D vectors)
            let attn_norm_w = dequantize(
                layer.attn_norm.data,
                layer.attn_norm.info.dtype,
                layer.attn_norm.info.n_elements(),
            );

            // 2a. Attention norm
            let h = if use_gpu {
                self.backend.rms_norm(&x, &attn_norm_w, cfg.norm_eps)
            } else {
                self.cpu_backend.rms_norm(&x, &attn_norm_w, cfg.norm_eps)
            };

            // 2b. Q, K, V projections
            let mut q = if use_gpu {
                self.backend.matmul_vec(&layer.wq, &h)
            } else {
                self.cpu_backend.matmul_vec(&layer.wq, &h)
            };
            let mut k = if use_gpu {
                self.backend.matmul_vec(&layer.wk, &h)
            } else {
                self.cpu_backend.matmul_vec(&layer.wk, &h)
            };
            let v = if use_gpu {
                self.backend.matmul_vec(&layer.wv, &h)
            } else {
                self.cpu_backend.matmul_vec(&layer.wv, &h)
            };

            // 2c. Apply RoPE
            if use_gpu {
                self.backend
                    .rope(&mut q, &mut k, pos, head_dim, cfg.rope_freq_base);
            } else {
                self.cpu_backend
                    .rope(&mut q, &mut k, pos, head_dim, cfg.rope_freq_base);
            }

            // 2d. Store K, V into cache
            let kv_offset = pos * kv_dim;
            self.kv_cache.key_cache[layer_idx][kv_offset..kv_offset + kv_dim].copy_from_slice(&k);
            self.kv_cache.val_cache[layer_idx][kv_offset..kv_offset + kv_dim].copy_from_slice(&v);
            if use_gpu {
                self.backend.store_kv(layer_idx, pos, &k, &v, kv_dim);
            }

            // 2e. Multi-head attention with GQA
            let seq_len = pos + 1;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let attn_out = if use_gpu {
                self.backend.attention_layer(
                    layer_idx,
                    &q,
                    &self.kv_cache.key_cache[layer_idx],
                    &self.kv_cache.val_cache[layer_idx],
                    cfg.n_heads,
                    n_heads_per_kv,
                    head_dim,
                    seq_len,
                    kv_dim,
                    scale,
                )
            } else {
                self.cpu_backend.attention_layer(
                    layer_idx,
                    &q,
                    &self.kv_cache.key_cache[layer_idx],
                    &self.kv_cache.val_cache[layer_idx],
                    cfg.n_heads,
                    n_heads_per_kv,
                    head_dim,
                    seq_len,
                    kv_dim,
                    scale,
                )
            };

            // 2f. Output projection + residual
            let attn_proj = if use_gpu {
                self.backend.matmul_vec(&layer.wo, &attn_out)
            } else {
                self.cpu_backend.matmul_vec(&layer.wo, &attn_out)
            };
            x = if use_gpu {
                self.backend.add(&x, &attn_proj)
            } else {
                self.cpu_backend.add(&x, &attn_proj)
            };

            // 2g. FFN norm
            let ffn_norm_w = dequantize(
                layer.ffn_norm.data,
                layer.ffn_norm.info.dtype,
                layer.ffn_norm.info.n_elements(),
            );
            let h = if use_gpu {
                self.backend.rms_norm(&x, &ffn_norm_w, cfg.norm_eps)
            } else {
                self.cpu_backend.rms_norm(&x, &ffn_norm_w, cfg.norm_eps)
            };

            // 2h. SwiGLU FFN: x = x + w2 @ (silu(w1 @ h) * (w3 @ h))
            let gate = if use_gpu {
                self.backend.matmul_vec(&layer.w1, &h)
            } else {
                self.cpu_backend.matmul_vec(&layer.w1, &h)
            };
            let up = if use_gpu {
                self.backend.matmul_vec(&layer.w3, &h)
            } else {
                self.cpu_backend.matmul_vec(&layer.w3, &h)
            };
            let gate_act = if use_gpu {
                self.backend.silu(&gate)
            } else {
                self.cpu_backend.silu(&gate)
            };
            let ffn_hidden = if use_gpu {
                self.backend.mul(&gate_act, &up)
            } else {
                self.cpu_backend.mul(&gate_act, &up)
            };
            let ffn_out = if use_gpu {
                self.backend.matmul_vec(&layer.w2, &ffn_hidden)
            } else {
                self.cpu_backend.matmul_vec(&layer.w2, &ffn_hidden)
            };

            x = if use_gpu {
                self.backend.add(&x, &ffn_out)
            } else {
                self.cpu_backend.add(&x, &ffn_out)
            };
        }

        // 3. Final norm
        let final_use_gpu = cfg.n_layers > 0
            && self
                .backend
                .should_use_layer_gpu(cfg.n_layers - 1, cfg.n_layers);
        let output_norm_w = dequantize(
            w.output_norm.data,
            w.output_norm.info.dtype,
            w.output_norm.info.n_elements(),
        );
        x = if final_use_gpu {
            self.backend.rms_norm(&x, &output_norm_w, cfg.norm_eps)
        } else {
            self.cpu_backend.rms_norm(&x, &output_norm_w, cfg.norm_eps)
        };

        // 4. Output projection → logits [vocab_size]
        let logits = if final_use_gpu {
            self.backend.matmul_vec(&w.output, &x)
        } else {
            self.cpu_backend.matmul_vec(&w.output, &x)
        };
        Ok(logits)
    }
}
