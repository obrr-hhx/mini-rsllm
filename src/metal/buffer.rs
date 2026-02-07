use crate::gguf::GgufDType;
use crate::tensor::dequantize;

#[derive(Debug, Clone, PartialEq)]
pub struct BufferMeta {
    pub shape: Vec<usize>,
    pub dtype: GgufDType,
    pub n_elements: usize,
}

#[derive(Debug, Clone)]
pub struct MetalBuffer {
    bytes: Vec<u8>,
    meta: BufferMeta,
}

impl MetalBuffer {
    pub fn from_bytes(bytes: Vec<u8>, shape: Vec<usize>, dtype: GgufDType) -> Result<Self, String> {
        let n_elements = shape.iter().product::<usize>();
        let (block_bytes, block_elems) = dtype.block_size();
        if n_elements % block_elems != 0 {
            return Err(format!(
                "invalid element count {} for dtype {:?} (block size {})",
                n_elements, dtype, block_elems
            ));
        }
        let expected = (n_elements / block_elems) * block_bytes;
        if bytes.len() != expected {
            return Err(format!(
                "invalid byte length {} for dtype {:?}: expected {}",
                bytes.len(),
                dtype,
                expected
            ));
        }
        Ok(MetalBuffer {
            bytes,
            meta: BufferMeta {
                shape,
                dtype,
                n_elements,
            },
        })
    }

    pub fn from_f32_slice(slice: &[f32], shape: Vec<usize>) -> Result<Self, String> {
        if shape.iter().product::<usize>() != slice.len() {
            return Err(format!(
                "shape {:?} does not match data length {}",
                shape,
                slice.len()
            ));
        }
        let mut bytes = Vec::with_capacity(slice.len() * std::mem::size_of::<f32>());
        for v in slice {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_bytes(bytes, shape, GgufDType::F32)
    }

    pub fn meta(&self) -> &BufferMeta {
        &self.meta
    }

    pub fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn upload_from_host(&mut self, src: &[u8]) -> Result<(), String> {
        if src.len() != self.bytes.len() {
            return Err(format!(
                "upload length mismatch: got {}, expected {}",
                src.len(),
                self.bytes.len()
            ));
        }
        self.bytes.copy_from_slice(src);
        Ok(())
    }

    pub fn download_to_host(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    pub fn download_f32(&self) -> Vec<f32> {
        dequantize(&self.bytes, self.meta.dtype, self.meta.n_elements)
    }
}

#[cfg(test)]
mod tests {
    use super::MetalBuffer;
    use crate::gguf::GgufDType;
    use half::f16;

    #[test]
    fn f32_roundtrip_with_shape_and_meta() {
        let data = vec![1.0f32, -2.0f32, 0.5f32, 4.0f32];
        let shape = vec![2usize, 2usize];
        let buf = MetalBuffer::from_f32_slice(&data, shape.clone()).unwrap();

        assert_eq!(buf.meta().shape, shape);
        assert_eq!(buf.meta().dtype, GgufDType::F32);
        assert_eq!(buf.meta().n_elements, 4);
        assert_eq!(buf.download_f32(), data);
    }

    #[test]
    fn host_upload_download_keeps_q4_layout() {
        // One Q4_0 block (32 elements): [scale_f16][16 packed nibbles]
        let mut bytes = Vec::with_capacity(18);
        bytes.extend_from_slice(&f16::from_f32(0.5f32).to_le_bytes());
        for i in 0..16u8 {
            bytes.push((i & 0x0F) | (((15u8 - i) & 0x0F) << 4));
        }

        let mut buf =
            MetalBuffer::from_bytes(bytes.clone(), vec![32usize], GgufDType::Q4_0).unwrap();
        let roundtrip = buf.download_to_host();
        assert_eq!(roundtrip, bytes);

        let mut replaced = bytes.clone();
        replaced[2] = 0x21;
        buf.upload_from_host(&replaced).unwrap();
        assert_eq!(buf.download_to_host(), replaced);
        assert_eq!(buf.download_f32().len(), 32);
    }

    #[test]
    fn invalid_length_is_rejected() {
        let bad = vec![0u8; 17];
        let err = MetalBuffer::from_bytes(bad, vec![32usize], GgufDType::Q4_0).unwrap_err();
        assert!(err.contains("invalid byte length"));
    }
}
