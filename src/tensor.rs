// Tensor storage, dequantization, and math operations
// All computation happens in f32. Quantized weights are dequantized on-the-fly.

use crate::gguf::{GgufDType, TensorInfo};
use half::f16;

/// A simple tensor: contiguous f32 data with a shape.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Tensor { data, shape }
    }
}

// --- Dequantization: convert raw bytes → Vec<f32> ---

/// Dequantize raw bytes into f32 values based on the tensor's dtype.
pub fn dequantize(data: &[u8], dtype: GgufDType, n_elements: usize) -> Vec<f32> {
    match dtype {
        GgufDType::F32 => dequantize_f32(data, n_elements),
        GgufDType::F16 => dequantize_f16(data, n_elements),
        GgufDType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgufDType::Q4_0 => dequantize_q4_0(data, n_elements),
        GgufDType::Q6_K => dequantize_q6_k(data, n_elements),
    }
}

fn dequantize_f32(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
        out.push(f32::from_le_bytes(bytes));
    }
    out
}

fn dequantize_f16(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = [data[i * 2], data[i * 2 + 1]];
        out.push(f16::from_le_bytes(bytes).to_f32());
    }
    out
}

/// Q8_0: blocks of 34 bytes = 1 f16 scale + 32 i8 quantized values
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);
    for block in 0..n_blocks {
        let offset = block * 34;
        let scale = f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
        for i in 0..32 {
            let quant = data[offset + 2 + i] as i8;
            out.push(scale * quant as f32);
        }
    }
    out
}

/// Q4_0: blocks of 18 bytes = 1 f16 scale + 16 bytes (32 4-bit values packed in pairs)
/// Layout: low nibbles fill positions 0-15, high nibbles fill positions 16-31.
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);
    for block in 0..n_blocks {
        let offset = block * 18;
        let scale = f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
        // First half: low nibbles (positions 0-15)
        for i in 0..16 {
            let byte = data[offset + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            out.push(scale * lo as f32);
        }
        // Second half: high nibbles (positions 16-31)
        for i in 0..16 {
            let byte = data[offset + 2 + i];
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out.push(scale * hi as f32);
        }
    }
    out
}

/// Q6_K: blocks of 210 bytes = ql[128] + qh[64] + scales[16] + d[2] = 256 values
/// Each value is 6 bits: lower 4 from ql, upper 2 from qh, biased by -32.
/// Follows the ggml reference dequantize_row_q6_K layout exactly.
fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 256;
    let mut out = Vec::with_capacity(n_elements);
        for block in 0..n_blocks {
            let bo = block * 210;
            let ql = &data[bo..bo + 128];
            let qh = &data[bo + 128..bo + 192];
            let scales = &data[bo + 192..bo + 208];
            let d = f16::from_le_bytes([data[bo + 208], data[bo + 209]]).to_f32();

            let mut block_out = [0.0f32; 256];
            // Process two 128-value halves
            for half in 0..2usize {
                let ql_off = half * 64;
                let qh_off = half * 32;
                let sc_base = half * 8;
                let y_off = half * 128;

                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                    let sc0 = scales[sc_base + is + 0] as i8 as f32;
                    let sc2 = scales[sc_base + is + 2] as i8 as f32;
                    let sc4 = scales[sc_base + is + 4] as i8 as f32;
                    let sc6 = scales[sc_base + is + 6] as i8 as f32;

                    block_out[y_off + l] = d * sc0 * q1 as f32;
                    block_out[y_off + l + 32] = d * sc2 * q2 as f32;
                    block_out[y_off + l + 64] = d * sc4 * q3 as f32;
                block_out[y_off + l + 96] = d * sc6 * q4 as f32;
            }
        }
        out.extend_from_slice(&block_out);
    }
    out
}

// --- Quantized dot product (on-the-fly dequantization) ---

/// Reference to a weight tensor stored in the GGUF mmap.
/// Holds raw bytes + metadata so we can dequantize on-the-fly.
pub struct TensorRef<'a> {
    pub data: &'a [u8],
    pub info: TensorInfo,
}

impl<'a> TensorRef<'a> {
    /// Number of rows (first dimension for 2D, or 1 for 1D).
    pub fn rows(&self) -> usize {
        if self.info.shape.len() >= 2 {
            self.info.shape[1]
        } else {
            1
        }
    }

    /// Number of columns (last dimension).
    pub fn cols(&self) -> usize {
        self.info.shape[0]
    }

    /// Dequantize the entire tensor into f32.
    pub fn to_tensor(&self) -> Tensor {
        let data = dequantize(self.data, self.info.dtype, self.info.n_elements());
        Tensor::from_vec(data, self.info.shape.clone())
    }

    /// Dequantize a single row of a 2D tensor.
    pub fn dequantize_row(&self, row: usize) -> Vec<f32> {
        let cols = self.cols();
        let (block_bytes, block_elems) = self.info.dtype.block_size();
        let row_bytes = (cols / block_elems) * block_bytes;
        let offset = row * row_bytes;
        let row_data = &self.data[offset..offset + row_bytes];
        dequantize(row_data, self.info.dtype, cols)
    }
}

/// Matrix-vector multiply: mat[rows, cols] @ vec[cols] -> out[rows]
/// Dequantizes each row on-the-fly to avoid materializing the full matrix.
pub fn matmul_vec(mat: &TensorRef, vec: &[f32]) -> Vec<f32> {
    let rows = mat.rows();
    let cols = mat.cols();
    debug_assert_eq!(vec.len(), cols);

    let mut out = Vec::with_capacity(rows);
    let (block_bytes, block_elems) = mat.info.dtype.block_size();
    let row_bytes = (cols / block_elems) * block_bytes;

    for row in 0..rows {
        let offset = row * row_bytes;
        let row_data = &mat.data[offset..offset + row_bytes];
        let dot = dot_product_quantized(row_data, vec, mat.info.dtype, cols);
        out.push(dot);
    }
    out
}

/// Compute dot product between a quantized row and an f32 vector.
/// Dequantizes block-by-block to keep memory usage minimal.
fn dot_product_quantized(row_data: &[u8], vec: &[f32], dtype: GgufDType, n: usize) -> f32 {
    match dtype {
        GgufDType::F32 => {
            let mut sum = 0.0f32;
            for i in 0..n {
                let bytes = [
                    row_data[i * 4],
                    row_data[i * 4 + 1],
                    row_data[i * 4 + 2],
                    row_data[i * 4 + 3],
                ];
                sum += f32::from_le_bytes(bytes) * vec[i];
            }
            sum
        }
        GgufDType::F16 => {
            let mut sum = 0.0f32;
            for i in 0..n {
                let bytes = [row_data[i * 2], row_data[i * 2 + 1]];
                let val = f16::from_le_bytes(bytes).to_f32();
                sum += val * vec[i];
            }
            sum
        }
        GgufDType::Q8_0 => {
            let n_blocks = n / 32;
            let mut sum = 0.0f32;
            for block in 0..n_blocks {
                let bo = block * 34;
                let scale = f16::from_le_bytes([row_data[bo], row_data[bo + 1]]).to_f32();
                let vi = block * 32;
                let mut block_sum = 0.0f32;
                for i in 0..32 {
                    let quant = row_data[bo + 2 + i] as i8;
                    block_sum += quant as f32 * vec[vi + i];
                }
                sum += scale * block_sum;
            }
            sum
        }
        GgufDType::Q4_0 => {
            let n_blocks = n / 32;
            let mut sum = 0.0f32;
            for block in 0..n_blocks {
                let bo = block * 18;
                let scale = f16::from_le_bytes([row_data[bo], row_data[bo + 1]]).to_f32();
                let vi = block * 32;
                let mut block_sum = 0.0f32;
                for i in 0..16 {
                    let byte = row_data[bo + 2 + i];
                    let lo = (byte & 0x0F) as i32 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                    block_sum += lo as f32 * vec[vi + i];      // positions 0-15
                    block_sum += hi as f32 * vec[vi + 16 + i]; // positions 16-31
                }
                sum += scale * block_sum;
            }
            sum
        }
        GgufDType::Q6_K => {
            let n_blocks = n / 256;
            let mut sum = 0.0f32;
            for block in 0..n_blocks {
                let bo = block * 210;
                let ql = &row_data[bo..bo + 128];
                let qh = &row_data[bo + 128..bo + 192];
                let scales = &row_data[bo + 192..bo + 208];
                let d = f16::from_le_bytes([row_data[bo + 208], row_data[bo + 209]]).to_f32();
                let vi = block * 256;

                for half in 0..2usize {
                    let ql_off = half * 64;
                    let qh_off = half * 32;
                    let sc_base = half * 8;
                    let y_off = half * 128;

                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                        let sc0 = scales[sc_base + is + 0] as i8 as f32;
                        let sc2 = scales[sc_base + is + 2] as i8 as f32;
                        let sc4 = scales[sc_base + is + 4] as i8 as f32;
                        let sc6 = scales[sc_base + is + 6] as i8 as f32;

                        sum += d * sc0 * q1 as f32 * vec[vi + y_off + l];
                        sum += d * sc2 * q2 as f32 * vec[vi + y_off + l + 32];
                        sum += d * sc4 * q3 as f32 * vec[vi + y_off + l + 64];
                        sum += d * sc6 * q4 as f32 * vec[vi + y_off + l + 96];
                    }
                }
            }
            sum
        }
    }
}

// --- Tensor operations ---

/// RMS normalization: out[i] = (x[i] / rms(x)) * weight[i]
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let rms = (ss / n as f32 + eps).sqrt();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((x[i] / rms) * weight[i]);
    }
    out
}

/// Numerically stable softmax (in-place).
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| v * (1.0 / (1.0 + (-v).exp())))
        .collect()
}

/// Apply Rotary Position Embedding to q and k vectors in-place.
/// q and k are [n_heads * head_dim] shaped, applied per head.
pub fn rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let n_q_heads = q.len() / head_dim;
    let n_k_heads = k.len() / head_dim;

    for h in 0..n_q_heads {
        let offset = h * head_dim;
        apply_rope_to_head(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
    }
    for h in 0..n_k_heads {
        let offset = h * head_dim;
        apply_rope_to_head(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
    }
}

fn apply_rope_to_head(head: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / freq_base.powf(i as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let x0 = head[i];
        let x1 = head[i + 1];
        head[i] = x0 * cos_t - x1 * sin_t;
        head[i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

/// Element-wise addition: a + b
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Element-wise multiplication: a * b
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}
