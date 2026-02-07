use std::ffi::{c_char, CStr};

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn rsllm_metal_is_available(err: *mut *mut c_char) -> i32;
    fn rsllm_metal_matvec_f32(
        matrix: *const f32,
        vec: *const f32,
        out: *mut f32,
        rows: u32,
        cols: u32,
        err: *mut *mut c_char,
    ) -> i32;
    fn rsllm_metal_matvec_f16(
        matrix: *const u16,
        vec: *const f32,
        out: *mut f32,
        rows: u32,
        cols: u32,
        err: *mut *mut c_char,
    ) -> i32;
    fn rsllm_metal_matvec_q4_0(
        matrix: *const u8,
        vec: *const f32,
        out: *mut f32,
        rows: u32,
        cols: u32,
        err: *mut *mut c_char,
    ) -> i32;
    fn rsllm_metal_rms_norm(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        n: u32,
        eps: f32,
        err: *mut *mut c_char,
    ) -> i32;
    fn rsllm_metal_apply_rope(
        input: *const f32,
        out: *mut f32,
        len: u32,
        pos: u32,
        head_dim: u32,
        freq_base: f32,
        err: *mut *mut c_char,
    ) -> i32;
    fn rsllm_metal_softmax(input: *const f32, out: *mut f32, n: u32, err: *mut *mut c_char) -> i32;
    fn rsllm_metal_free_error(err: *mut c_char);
}

#[derive(Debug)]
pub struct MetalContext;

impl MetalContext {
    pub fn new(_gpu_layers: usize) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let ok = unsafe { rsllm_metal_is_available(&mut err_ptr) };
            if ok == 0 {
                return Err(take_error_message(err_ptr));
            }
        }
        Ok(MetalContext)
    }

    pub fn matvec_f32(
        &self,
        matrix_bytes: &[u8],
        rows: usize,
        cols: usize,
        vec: &[f32],
    ) -> Result<Vec<f32>, String> {
        if vec.len() != cols {
            return Err(format!(
                "matvec_f32 shape mismatch: vec len {} != cols {}",
                vec.len(),
                cols
            ));
        }
        let expected = rows
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "matvec_f32 size overflow".to_string())?;
        if matrix_bytes.len() != expected {
            return Err(format!(
                "matvec_f32 matrix bytes mismatch: got {}, expected {}",
                matrix_bytes.len(),
                expected
            ));
        }
        if rows == 0 {
            return Ok(Vec::new());
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0f32; rows];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_matvec_f32(
                    matrix_bytes.as_ptr() as *const f32,
                    vec.as_ptr(),
                    out.as_mut_ptr(),
                    rows as u32,
                    cols as u32,
                    &mut err_ptr,
                )
            };
            if status == 0 {
                Ok(out)
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = matrix_bytes;
            let _ = rows;
            let _ = cols;
            let _ = vec;
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn matvec_f16(
        &self,
        matrix_bytes: &[u8],
        rows: usize,
        cols: usize,
        vec: &[f32],
    ) -> Result<Vec<f32>, String> {
        if vec.len() != cols {
            return Err(format!(
                "matvec_f16 shape mismatch: vec len {} != cols {}",
                vec.len(),
                cols
            ));
        }
        let expected = rows
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "matvec_f16 size overflow".to_string())?;
        if matrix_bytes.len() != expected {
            return Err(format!(
                "matvec_f16 matrix bytes mismatch: got {}, expected {}",
                matrix_bytes.len(),
                expected
            ));
        }
        if rows == 0 {
            return Ok(Vec::new());
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0f32; rows];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_matvec_f16(
                    matrix_bytes.as_ptr() as *const u16,
                    vec.as_ptr(),
                    out.as_mut_ptr(),
                    rows as u32,
                    cols as u32,
                    &mut err_ptr,
                )
            };
            if status == 0 {
                Ok(out)
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = matrix_bytes;
            let _ = rows;
            let _ = cols;
            let _ = vec;
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn matvec_q4_0(
        &self,
        matrix_bytes: &[u8],
        rows: usize,
        cols: usize,
        vec: &[f32],
    ) -> Result<Vec<f32>, String> {
        if vec.len() != cols {
            return Err(format!(
                "matvec_q4_0 shape mismatch: vec len {} != cols {}",
                vec.len(),
                cols
            ));
        }
        if cols % 32 != 0 {
            return Err(format!("matvec_q4_0 requires cols % 32 == 0, got {}", cols));
        }
        let row_bytes = (cols / 32)
            .checked_mul(18)
            .ok_or_else(|| "matvec_q4_0 row_bytes overflow".to_string())?;
        let expected = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| "matvec_q4_0 size overflow".to_string())?;
        if matrix_bytes.len() != expected {
            return Err(format!(
                "matvec_q4_0 matrix bytes mismatch: got {}, expected {}",
                matrix_bytes.len(),
                expected
            ));
        }
        if rows == 0 {
            return Ok(Vec::new());
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0f32; rows];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_matvec_q4_0(
                    matrix_bytes.as_ptr(),
                    vec.as_ptr(),
                    out.as_mut_ptr(),
                    rows as u32,
                    cols as u32,
                    &mut err_ptr,
                )
            };
            if status == 0 {
                Ok(out)
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = matrix_bytes;
            let _ = rows;
            let _ = cols;
            let _ = vec;
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, String> {
        if x.len() != weight.len() {
            return Err(format!(
                "rms_norm shape mismatch: x len {} != weight len {}",
                x.len(),
                weight.len()
            ));
        }
        if x.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0f32; x.len()];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_rms_norm(
                    x.as_ptr(),
                    weight.as_ptr(),
                    out.as_mut_ptr(),
                    x.len() as u32,
                    eps,
                    &mut err_ptr,
                )
            };
            if status == 0 {
                Ok(out)
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = x;
            let _ = weight;
            let _ = eps;
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn apply_rope(
        &self,
        data: &mut [f32],
        pos: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(format!(
                "apply_rope requires non-zero even head_dim, got {}",
                head_dim
            ));
        }
        if data.len() % head_dim != 0 {
            return Err(format!(
                "apply_rope requires data len % head_dim == 0, got len {} and head_dim {}",
                data.len(),
                head_dim
            ));
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0f32; data.len()];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_apply_rope(
                    data.as_ptr(),
                    out.as_mut_ptr(),
                    data.len() as u32,
                    pos as u32,
                    head_dim as u32,
                    freq_base,
                    &mut err_ptr,
                )
            };
            if status == 0 {
                data.copy_from_slice(&out);
                Ok(())
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = data;
            let _ = pos;
            let _ = head_dim;
            let _ = freq_base;
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn softmax(&self, x: &mut [f32]) -> Result<(), String> {
        if x.is_empty() {
            return Ok(());
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0f32; x.len()];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_softmax(x.as_ptr(), out.as_mut_ptr(), x.len() as u32, &mut err_ptr)
            };
            if status == 0 {
                x.copy_from_slice(&out);
                Ok(())
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = x;
            Err("Metal backend is only available on macOS".to_string())
        }
    }
}

#[cfg(target_os = "macos")]
fn take_error_message(err_ptr: *mut c_char) -> String {
    if err_ptr.is_null() {
        return "unknown Metal bridge error".to_string();
    }
    let msg = unsafe { CStr::from_ptr(err_ptr) }
        .to_string_lossy()
        .to_string();
    unsafe { rsllm_metal_free_error(err_ptr) };
    msg
}

#[cfg(not(target_os = "macos"))]
fn take_error_message(_err_ptr: *mut c_char) -> String {
    "Metal backend is only available on macOS".to_string()
}

#[cfg(test)]
mod tests {
    use crate::tensor;
    use half::f16;

    use super::MetalContext;

    fn cpu_matvec(mat: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += mat[r * cols + c] * vec[c];
            }
            out[r] = sum;
        }
        out
    }

    #[test]
    fn matvec_f32_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let rows = 3usize;
        let cols = 4usize;
        let mat: Vec<f32> = vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, -1.0f32, 0.5f32, 2.0f32, 1.0f32, 0.0f32, 1.0f32,
            0.0f32, -1.0f32,
        ];
        let vec: Vec<f32> = vec![0.25f32, -0.5f32, 1.0f32, 2.0f32];

        let mut bytes = Vec::with_capacity(mat.len() * std::mem::size_of::<f32>());
        for v in &mat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let gpu = ctx.matvec_f32(&bytes, rows, cols, &vec).unwrap();
        let cpu = cpu_matvec(&mat, rows, cols, &vec);

        for i in 0..rows {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-4,
                "row {} mismatch: gpu={}, cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn matvec_f16_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let rows = 2usize;
        let cols = 4usize;
        let mat_f32: Vec<f32> = vec![
            1.0f32, -2.0f32, 0.5f32, 4.0f32, -0.5f32, 1.5f32, 2.0f32, -1.0f32,
        ];
        let vec: Vec<f32> = vec![1.0f32, 0.25f32, -0.5f32, 2.0f32];

        let mut bytes = Vec::with_capacity(mat_f32.len() * std::mem::size_of::<u16>());
        for v in &mat_f32 {
            bytes.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }

        let gpu = ctx.matvec_f16(&bytes, rows, cols, &vec).unwrap();
        let cpu = cpu_matvec(&mat_f32, rows, cols, &vec);

        for i in 0..rows {
            assert!(
                (gpu[i] - cpu[i]).abs() < 2e-3,
                "row {} mismatch: gpu={}, cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn matvec_q4_0_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let rows = 2usize;
        let cols = 32usize;
        let vec: Vec<f32> = (0..cols).map(|i| (i as f32 - 16.0f32) * 0.125f32).collect();

        // Two Q4_0 rows (18 bytes each): [scale_f16_le][16 packed nibbles]
        let mut matrix = Vec::with_capacity(rows * 18);
        let scale0 = f16::from_f32(0.25f32).to_le_bytes();
        matrix.extend_from_slice(&scale0);
        for i in 0..16u8 {
            // lo = i%16, hi = (15-i)%16
            matrix.push((i & 0x0F) | (((15u8 - i) & 0x0F) << 4));
        }

        let scale1 = f16::from_f32(0.5f32).to_le_bytes();
        matrix.extend_from_slice(&scale1);
        for i in 0..16u8 {
            // deterministic different pattern
            let lo = (3u8 * i + 1u8) & 0x0F;
            let hi = (11u8 - i) & 0x0F;
            matrix.push(lo | (hi << 4));
        }

        let gpu = ctx.matvec_q4_0(&matrix, rows, cols, &vec).unwrap();

        // CPU reference decode and dot
        let mut cpu = vec![0.0f32; rows];
        for r in 0..rows {
            let bo = r * 18;
            let scale = f16::from_le_bytes([matrix[bo], matrix[bo + 1]]).to_f32();
            let mut block_sum = 0.0f32;
            for i in 0..16usize {
                let packed = matrix[bo + 2 + i];
                let lo = (packed & 0x0F) as i32 - 8;
                let hi = ((packed >> 4) & 0x0F) as i32 - 8;
                block_sum += lo as f32 * vec[i];
                block_sum += hi as f32 * vec[16 + i];
            }
            cpu[r] = scale * block_sum;
        }

        for i in 0..rows {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-3,
                "row {} mismatch: gpu={}, cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn rms_norm_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let x = vec![1.0f32, -2.0f32, 0.5f32, 4.0f32, -1.25f32, 0.25f32];
        let w = vec![1.1f32, 0.9f32, 1.0f32, 0.8f32, 1.2f32, 0.7f32];
        let eps = 1e-5f32;

        let gpu = ctx.rms_norm(&x, &w, eps).unwrap();
        let cpu = tensor::rms_norm(&x, &w, eps);

        for i in 0..x.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-5,
                "idx {} mismatch: gpu={}, cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn rope_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let pos = 7usize;
        let head_dim = 4usize;
        let freq_base = 10000.0f32;

        let mut q_cpu = vec![
            0.2f32, -0.1f32, 0.4f32, 0.5f32, -0.3f32, 0.7f32, 0.8f32, -0.2f32,
        ];
        let mut k_cpu = vec![0.15f32, 0.25f32, -0.35f32, 0.45f32];
        tensor::rope(&mut q_cpu, &mut k_cpu, pos, head_dim, freq_base);

        let mut q_gpu = vec![
            0.2f32, -0.1f32, 0.4f32, 0.5f32, -0.3f32, 0.7f32, 0.8f32, -0.2f32,
        ];
        let mut k_gpu = vec![0.15f32, 0.25f32, -0.35f32, 0.45f32];
        ctx.apply_rope(&mut q_gpu, pos, head_dim, freq_base)
            .unwrap();
        ctx.apply_rope(&mut k_gpu, pos, head_dim, freq_base)
            .unwrap();

        for i in 0..q_gpu.len() {
            assert!(
                (q_gpu[i] - q_cpu[i]).abs() < 1e-5,
                "q idx {} mismatch: gpu={}, cpu={}",
                i,
                q_gpu[i],
                q_cpu[i]
            );
        }
        for i in 0..k_gpu.len() {
            assert!(
                (k_gpu[i] - k_cpu[i]).abs() < 1e-5,
                "k idx {} mismatch: gpu={}, cpu={}",
                i,
                k_gpu[i],
                k_cpu[i]
            );
        }
    }

    #[test]
    fn softmax_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let mut gpu = vec![1.0f32, -2.0f32, 0.5f32, 4.0f32, -1.25f32];
        let mut cpu = gpu.clone();
        ctx.softmax(&mut gpu).unwrap();
        tensor::softmax(&mut cpu);

        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-5,
                "idx {} mismatch: gpu={}, cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
    }
}
