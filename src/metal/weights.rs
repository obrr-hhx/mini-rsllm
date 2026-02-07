use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use crate::gguf::{GgufDType, TensorInfo};

#[derive(Debug, Clone)]
pub struct MetalWeightStats {
    pub uploaded_tensors: usize,
    pub uploaded_bytes: usize,
}

#[derive(Debug, Clone)]
struct WeightRecord {
    shape: Vec<usize>,
    dtype: GgufDType,
    bytes: usize,
    fingerprint: u64,
}

#[derive(Debug, Default)]
struct Inner {
    records: HashMap<String, WeightRecord>,
    uploaded_tensors: usize,
    uploaded_bytes: usize,
}

#[derive(Debug, Default)]
pub struct MetalWeightStore {
    inner: Mutex<Inner>,
}

impl MetalWeightStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tensor as uploaded. Returns true only for the first upload.
    /// A repeated call with the same tensor name and fingerprint is treated as cache hit.
    pub fn register_tensor(&self, info: &TensorInfo, data: &[u8]) -> bool {
        let fp = fingerprint(data);
        let mut inner = self.inner.lock().expect("weight store lock poisoned");
        if let Some(existing) = inner.records.get(&info.name) {
            if existing.fingerprint == fp
                && existing.bytes == data.len()
                && existing.dtype == info.dtype
                && existing.shape == info.shape
            {
                return false;
            }
        }

        inner.records.insert(
            info.name.clone(),
            WeightRecord {
                shape: info.shape.clone(),
                dtype: info.dtype,
                bytes: data.len(),
                fingerprint: fp,
            },
        );
        inner.uploaded_tensors += 1;
        inner.uploaded_bytes += data.len();
        true
    }

    pub fn stats(&self) -> MetalWeightStats {
        let inner = self.inner.lock().expect("weight store lock poisoned");
        MetalWeightStats {
            uploaded_tensors: inner.uploaded_tensors,
            uploaded_bytes: inner.uploaded_bytes,
        }
    }
}

fn fingerprint(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.len().hash(&mut hasher);
    let prefix_len = data.len().min(128);
    data[..prefix_len].hash(&mut hasher);
    if data.len() > 128 {
        data[data.len() - 128..].hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::MetalWeightStore;
    use crate::gguf::{GgufDType, TensorInfo};

    fn make_info(name: &str, shape: Vec<usize>, dtype: GgufDType) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape,
            dtype,
            offset: 0,
        }
    }

    #[test]
    fn register_same_tensor_only_once() {
        let store = MetalWeightStore::new();
        let info = make_info("w", vec![4usize], GgufDType::F32);
        let mut data = Vec::new();
        for v in [1.0f32, 2.0f32, 3.0f32, 4.0f32] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        assert!(store.register_tensor(&info, &data));
        assert!(!store.register_tensor(&info, &data));
        let stats = store.stats();
        assert_eq!(stats.uploaded_tensors, 1);
        assert_eq!(stats.uploaded_bytes, data.len());
    }

    #[test]
    fn changed_tensor_content_counts_as_new_upload() {
        let store = MetalWeightStore::new();
        let info = make_info("w", vec![4usize], GgufDType::F32);
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();
        for v in [1.0f32, 2.0f32, 3.0f32, 4.0f32] {
            data1.extend_from_slice(&v.to_le_bytes());
        }
        for v in [1.0f32, 2.0f32, 3.0f32, 5.0f32] {
            data2.extend_from_slice(&v.to_le_bytes());
        }

        assert!(store.register_tensor(&info, &data1));
        assert!(store.register_tensor(&info, &data2));
        let stats = store.stats();
        assert_eq!(stats.uploaded_tensors, 2);
        assert_eq!(stats.uploaded_bytes, data1.len() + data2.len());
    }
}
