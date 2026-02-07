#[derive(Debug, Default)]
pub struct MetalWeightStore {
    uploaded_tensors: usize,
}

impl MetalWeightStore {
    pub fn new() -> Self {
        MetalWeightStore {
            uploaded_tensors: 0,
        }
    }

    pub fn uploaded_tensors(&self) -> usize {
        self.uploaded_tensors
    }

    pub fn mark_uploaded(&mut self, count: usize) {
        self.uploaded_tensors += count;
    }
}
