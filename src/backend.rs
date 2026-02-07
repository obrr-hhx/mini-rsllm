// Backend abstraction for model math operators.
// CPU is fully implemented; Metal is feature-gated and currently falls back to CPU ops.

#[cfg(feature = "metal")]
use crate::gguf::GgufDType;
use crate::tensor;
use crate::tensor::TensorRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Metal,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Metal => "metal",
        }
    }
}

impl std::str::FromStr for DeviceKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(DeviceKind::Cpu),
            "metal" => Ok(DeviceKind::Metal),
            _ => Err(format!(
                "unsupported device '{}'; expected one of: cpu, metal",
                s
            )),
        }
    }
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    fn matmul_vec(&self, mat: &TensorRef, vec: &[f32]) -> Vec<f32>;
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32>;
    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freq_base: f32);
    fn softmax(&self, x: &mut [f32]);
    fn silu(&self, x: &[f32]) -> Vec<f32>;
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn store_kv(&self, _layer_idx: usize, _pos: usize, _k: &[f32], _v: &[f32], _kv_dim: usize) {}
    fn attention_head(
        &self,
        _layer_idx: usize,
        q_head: &[f32],
        key_cache_layer: &[f32],
        val_cache_layer: &[f32],
        seq_len: usize,
        kv_dim: usize,
        kv_head_offset: usize,
        scale: f32,
    ) -> Vec<f32> {
        let head_dim = q_head.len();
        let mut scores = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let k_base = t * kv_dim + kv_head_offset;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * key_cache_layer[k_base + d];
            }
            scores.push(dot * scale);
        }
        self.softmax(&mut scores);

        let mut out = vec![0.0f32; head_dim];
        for t in 0..seq_len {
            let v_base = t * kv_dim + kv_head_offset;
            let s = scores[t];
            for d in 0..head_dim {
                out[d] += s * val_cache_layer[v_base + d];
            }
        }
        out
    }

    /// Whether this backend should run the given transformer layer on GPU.
    /// `layer_idx` is in [0, n_layers).
    fn should_use_layer_gpu(&self, _layer_idx: usize, _n_layers: usize) -> bool {
        false
    }
}

pub(crate) fn should_offload_layer(gpu_layers: usize, layer_idx: usize, n_layers: usize) -> bool {
    if gpu_layers == 0 {
        return false;
    }
    let start = n_layers.saturating_sub(gpu_layers);
    layer_idx >= start && layer_idx < n_layers
}

#[derive(Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn matmul_vec(&self, mat: &TensorRef, vec: &[f32]) -> Vec<f32> {
        tensor::matmul_vec(mat, vec)
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        tensor::rms_norm(x, weight, eps)
    }

    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
        tensor::rope(q, k, pos, head_dim, freq_base);
    }

    fn softmax(&self, x: &mut [f32]) {
        tensor::softmax(x);
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        tensor::silu(x)
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        tensor::add(a, b)
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        tensor::mul(a, b)
    }
}

#[cfg(feature = "metal")]
use crate::metal::context::MetalContext;
#[cfg(feature = "metal")]
use crate::metal::weights::MetalWeightStore;

#[cfg(feature = "metal")]
pub struct MetalBackend {
    cpu_fallback: CpuBackend,
    context: MetalContext,
    _weights: MetalWeightStore,
    gpu_layers: usize,
}

#[cfg(feature = "metal")]
impl MetalBackend {
    pub fn new(gpu_layers: usize) -> Result<Self, String> {
        let context = MetalContext::new(gpu_layers)?;
        let weights = MetalWeightStore::new();
        Ok(MetalBackend {
            cpu_fallback: CpuBackend::new(),
            context,
            _weights: weights,
            gpu_layers,
        })
    }
}

#[cfg(feature = "metal")]
impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal (cpu fallback)"
    }

    fn matmul_vec(&self, mat: &TensorRef, vec: &[f32]) -> Vec<f32> {
        if self.gpu_layers == 0 {
            return self.cpu_fallback.matmul_vec(mat, vec);
        }

        let rows = mat.rows();
        let cols = mat.cols();

        let gpu_result = match mat.info.dtype {
            GgufDType::F32 => self.context.matvec_f32(mat.data, rows, cols, vec),
            GgufDType::F16 => self.context.matvec_f16(mat.data, rows, cols, vec),
            GgufDType::Q4_0 => self.context.matvec_q4_0(mat.data, rows, cols, vec),
            _ => return self.cpu_fallback.matmul_vec(mat, vec),
        };

        match gpu_result {
            Ok(out) => out,
            Err(_) => self.cpu_fallback.matmul_vec(mat, vec),
        }
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        if self.gpu_layers == 0 {
            return self.cpu_fallback.rms_norm(x, weight, eps);
        }
        match self.context.rms_norm(x, weight, eps) {
            Ok(out) => out,
            Err(_) => self.cpu_fallback.rms_norm(x, weight, eps),
        }
    }

    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
        if self.gpu_layers == 0 {
            self.cpu_fallback.rope(q, k, pos, head_dim, freq_base);
            return;
        }

        let q_orig = q.to_vec();
        let k_orig = k.to_vec();

        let q_ok = self.context.apply_rope(q, pos, head_dim, freq_base).is_ok();
        let k_ok = self.context.apply_rope(k, pos, head_dim, freq_base).is_ok();
        if !q_ok || !k_ok {
            q.copy_from_slice(&q_orig);
            k.copy_from_slice(&k_orig);
            self.cpu_fallback.rope(q, k, pos, head_dim, freq_base);
        }
    }

    fn softmax(&self, x: &mut [f32]) {
        if self.gpu_layers == 0 || self.context.softmax(x).is_err() {
            self.cpu_fallback.softmax(x);
        }
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        self.cpu_fallback.silu(x)
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.cpu_fallback.add(a, b)
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.cpu_fallback.mul(a, b)
    }

    fn store_kv(&self, layer_idx: usize, pos: usize, k: &[f32], v: &[f32], kv_dim: usize) {
        if self.gpu_layers == 0 {
            return;
        }
        let _ = self.context.kv_store(layer_idx, pos, k, v, kv_dim);
    }

    fn attention_head(
        &self,
        layer_idx: usize,
        q_head: &[f32],
        key_cache_layer: &[f32],
        val_cache_layer: &[f32],
        seq_len: usize,
        kv_dim: usize,
        kv_head_offset: usize,
        scale: f32,
    ) -> Vec<f32> {
        if self.gpu_layers == 0 {
            return self.cpu_fallback.attention_head(
                layer_idx,
                q_head,
                key_cache_layer,
                val_cache_layer,
                seq_len,
                kv_dim,
                kv_head_offset,
                scale,
            );
        }

        match self
            .context
            .attention_head(q_head, layer_idx, seq_len, kv_dim, kv_head_offset, scale)
        {
            Ok(out) => out,
            Err(_) => self.cpu_fallback.attention_head(
                layer_idx,
                q_head,
                key_cache_layer,
                val_cache_layer,
                seq_len,
                kv_dim,
                kv_head_offset,
                scale,
            ),
        }
    }

    fn should_use_layer_gpu(&self, layer_idx: usize, n_layers: usize) -> bool {
        should_offload_layer(self.gpu_layers, layer_idx, n_layers)
    }
}

pub fn build_backend(device: DeviceKind, gpu_layers: usize) -> Result<Box<dyn Backend>, String> {
    match device {
        DeviceKind::Cpu => Ok(Box::new(CpuBackend::new())),
        DeviceKind::Metal => {
            #[cfg(feature = "metal")]
            {
                let backend = MetalBackend::new(gpu_layers)?;
                Ok(Box::new(backend))
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = gpu_layers;
                Err("Metal backend is disabled. Rebuild with `--features metal`.".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{build_backend, should_offload_layer, DeviceKind};

    #[test]
    fn parse_device_kind() {
        assert_eq!("cpu".parse::<DeviceKind>().unwrap(), DeviceKind::Cpu);
        assert_eq!("metal".parse::<DeviceKind>().unwrap(), DeviceKind::Metal);
        assert!("cuda".parse::<DeviceKind>().is_err());
    }

    #[test]
    fn cpu_backend_factory_works() {
        let backend = build_backend(DeviceKind::Cpu, 0).unwrap();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn offload_policy_matches_gpu_layers() {
        assert!(!should_offload_layer(0, 0, 32));
        assert!(!should_offload_layer(4, 27, 32));
        assert!(should_offload_layer(4, 28, 32));
        assert!(should_offload_layer(4, 31, 32));
        assert!(should_offload_layer(64, 0, 32));
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn metal_backend_requires_feature() {
        let result = build_backend(DeviceKind::Metal, 1);
        assert!(result.is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_backend_factory_works() {
        let result = build_backend(DeviceKind::Metal, 1);
        if let Ok(backend) = result {
            assert_eq!(backend.name(), "metal (cpu fallback)");
        }
    }
}
