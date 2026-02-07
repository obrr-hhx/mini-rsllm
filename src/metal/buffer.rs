#[derive(Debug, Clone)]
pub struct MetalBuffer {
    data: Vec<f32>,
}

impl MetalBuffer {
    pub fn from_f32_slice(slice: &[f32]) -> Self {
        MetalBuffer {
            data: slice.to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}
