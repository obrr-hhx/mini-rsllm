// Backend abstraction for model math operators.
// CPU is fully implemented; Metal is feature-gated and currently falls back to CPU ops.

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
    _context: MetalContext,
    _weights: MetalWeightStore,
}

#[cfg(feature = "metal")]
impl MetalBackend {
    pub fn new(gpu_layers: usize) -> Result<Self, String> {
        let context = MetalContext::new(gpu_layers)?;
        let weights = MetalWeightStore::new();
        Ok(MetalBackend {
            cpu_fallback: CpuBackend::new(),
            _context: context,
            _weights: weights,
        })
    }
}

#[cfg(feature = "metal")]
impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal (cpu fallback)"
    }

    fn matmul_vec(&self, mat: &TensorRef, vec: &[f32]) -> Vec<f32> {
        self.cpu_fallback.matmul_vec(mat, vec)
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        self.cpu_fallback.rms_norm(x, weight, eps)
    }

    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
        self.cpu_fallback.rope(q, k, pos, head_dim, freq_base);
    }

    fn softmax(&self, x: &mut [f32]) {
        self.cpu_fallback.softmax(x);
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
    use super::{build_backend, DeviceKind};

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

    #[cfg(not(feature = "metal"))]
    #[test]
    fn metal_backend_requires_feature() {
        let result = build_backend(DeviceKind::Metal, 1);
        assert!(result.is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_backend_factory_works() {
        let backend = build_backend(DeviceKind::Metal, 1).unwrap();
        assert_eq!(backend.name(), "metal (cpu fallback)");
    }
}
