#[derive(Debug)]
pub struct MetalContext {
    gpu_layers: usize,
}

impl MetalContext {
    pub fn new(gpu_layers: usize) -> Result<Self, String> {
        Ok(MetalContext { gpu_layers })
    }

    pub fn gpu_layers(&self) -> usize {
        self.gpu_layers
    }
}
