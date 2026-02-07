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
    fn rsllm_metal_add_rms_norm(
        a: *const f32,
        b: *const f32,
        weight: *const f32,
        sum_out: *mut f32,
        norm_out: *mut f32,
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
    fn rsllm_metal_apply_rope_qk(
        q_in: *const f32,
        k_in: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        q_len: u32,
        k_len: u32,
        pos: u32,
        head_dim: u32,
        freq_base: f32,
        err: *mut *mut c_char,
    ) -> i32;
    fn rsllm_metal_kv_store(
        layer: u32,
        pos: u32,
        key: *const f32,
        val: *const f32,
        kv_dim: u32,
        err: *mut *mut c_char,
    ) -> i32;
    fn rsllm_metal_attn_head(
        q: *const f32,
        layer: u32,
        seq_len: u32,
        kv_dim: u32,
        kv_head_offset: u32,
        head_dim: u32,
        scale: f32,
        out: *mut f32,
        err: *mut *mut c_char,
    ) -> i32;
    fn rsllm_metal_attn_layer(
        q: *const f32,
        layer: u32,
        seq_len: u32,
        n_heads: u32,
        n_heads_per_kv: u32,
        head_dim: u32,
        kv_dim: u32,
        scale: f32,
        out: *mut f32,
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

    pub fn add_rms_norm(
        &self,
        a: &[f32],
        b: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if a.len() != b.len() || a.len() != weight.len() {
            return Err(format!(
                "add_rms_norm shape mismatch: a {}, b {}, weight {}",
                a.len(),
                b.len(),
                weight.len()
            ));
        }
        if a.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        #[cfg(target_os = "macos")]
        {
            let mut sum_out = vec![0.0f32; a.len()];
            let mut norm_out = vec![0.0f32; a.len()];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_add_rms_norm(
                    a.as_ptr(),
                    b.as_ptr(),
                    weight.as_ptr(),
                    sum_out.as_mut_ptr(),
                    norm_out.as_mut_ptr(),
                    a.len() as u32,
                    eps,
                    &mut err_ptr,
                )
            };
            if status == 0 {
                Ok((sum_out, norm_out))
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = a;
            let _ = b;
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

    pub fn apply_rope_qk(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> Result<(), String> {
        if q.is_empty() || k.is_empty() {
            return Ok(());
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(format!(
                "apply_rope_qk requires non-zero even head_dim, got {}",
                head_dim
            ));
        }
        if q.len() % head_dim != 0 || k.len() % head_dim != 0 {
            return Err(format!(
                "apply_rope_qk requires len % head_dim == 0, got q {} and k {} with head_dim {}",
                q.len(),
                k.len(),
                head_dim
            ));
        }

        #[cfg(target_os = "macos")]
        {
            let mut q_out = vec![0.0f32; q.len()];
            let mut k_out = vec![0.0f32; k.len()];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_apply_rope_qk(
                    q.as_ptr(),
                    k.as_ptr(),
                    q_out.as_mut_ptr(),
                    k_out.as_mut_ptr(),
                    q.len() as u32,
                    k.len() as u32,
                    pos as u32,
                    head_dim as u32,
                    freq_base,
                    &mut err_ptr,
                )
            };
            if status == 0 {
                q.copy_from_slice(&q_out);
                k.copy_from_slice(&k_out);
                Ok(())
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = q;
            let _ = k;
            let _ = pos;
            let _ = head_dim;
            let _ = freq_base;
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn kv_store(
        &self,
        layer_idx: usize,
        pos: usize,
        key: &[f32],
        val: &[f32],
        kv_dim: usize,
    ) -> Result<(), String> {
        if key.len() != kv_dim || val.len() != kv_dim {
            return Err(format!(
                "kv_store shape mismatch: key {}, val {}, kv_dim {}",
                key.len(),
                val.len(),
                kv_dim
            ));
        }

        #[cfg(target_os = "macos")]
        {
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_kv_store(
                    layer_idx as u32,
                    pos as u32,
                    key.as_ptr(),
                    val.as_ptr(),
                    kv_dim as u32,
                    &mut err_ptr,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(take_error_message(err_ptr))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = layer_idx;
            let _ = pos;
            let _ = key;
            let _ = val;
            let _ = kv_dim;
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn attention_head(
        &self,
        q_head: &[f32],
        layer_idx: usize,
        seq_len: usize,
        kv_dim: usize,
        kv_head_offset: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let head_dim = q_head.len();
        if head_dim == 0 {
            return Ok(Vec::new());
        }
        if seq_len == 0 {
            return Ok(vec![0.0f32; head_dim]);
        }
        if kv_head_offset + head_dim > kv_dim {
            return Err(format!(
                "attention_head range overflow: kv_head_offset {} + head_dim {} > kv_dim {}",
                kv_head_offset, head_dim, kv_dim
            ));
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0f32; head_dim];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_attn_head(
                    q_head.as_ptr(),
                    layer_idx as u32,
                    seq_len as u32,
                    kv_dim as u32,
                    kv_head_offset as u32,
                    head_dim as u32,
                    scale,
                    out.as_mut_ptr(),
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
            let _ = q_head;
            let _ = layer_idx;
            let _ = seq_len;
            let _ = kv_dim;
            let _ = kv_head_offset;
            let _ = scale;
            Err("Metal backend is only available on macOS".to_string())
        }
    }

    pub fn attention_layer(
        &self,
        q: &[f32],
        layer_idx: usize,
        seq_len: usize,
        n_heads: usize,
        n_heads_per_kv: usize,
        head_dim: usize,
        kv_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let expected = n_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "attention_layer size overflow".to_string())?;
        if q.len() != expected {
            return Err(format!(
                "attention_layer q shape mismatch: got {}, expected {}",
                q.len(),
                expected
            ));
        }
        if n_heads == 0 || head_dim == 0 {
            return Ok(Vec::new());
        }
        if n_heads_per_kv == 0 || n_heads % n_heads_per_kv != 0 {
            return Err(format!(
                "attention_layer invalid n_heads_per_kv: n_heads {}, n_heads_per_kv {}",
                n_heads, n_heads_per_kv
            ));
        }
        if kv_dim == 0 {
            return Err("attention_layer requires kv_dim > 0".to_string());
        }

        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0f32; expected];
            let mut err_ptr: *mut c_char = std::ptr::null_mut();
            let status = unsafe {
                rsllm_metal_attn_layer(
                    q.as_ptr(),
                    layer_idx as u32,
                    seq_len as u32,
                    n_heads as u32,
                    n_heads_per_kv as u32,
                    head_dim as u32,
                    kv_dim as u32,
                    scale,
                    out.as_mut_ptr(),
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
            let _ = q;
            let _ = layer_idx;
            let _ = seq_len;
            let _ = n_heads;
            let _ = n_heads_per_kv;
            let _ = head_dim;
            let _ = kv_dim;
            let _ = scale;
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
    fn add_rms_norm_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let a = vec![1.0f32, -2.0f32, 0.5f32, 4.0f32, -1.25f32, 0.25f32];
        let b = vec![-0.5f32, 0.75f32, 1.5f32, -0.25f32, 0.5f32, -1.0f32];
        let w = vec![1.1f32, 0.9f32, 1.0f32, 0.8f32, 1.2f32, 0.7f32];
        let eps = 1e-5f32;

        let (sum_gpu, norm_gpu) = ctx.add_rms_norm(&a, &b, &w, eps).unwrap();
        let sum_cpu = tensor::add(&a, &b);
        let norm_cpu = tensor::rms_norm(&sum_cpu, &w, eps);

        for i in 0..a.len() {
            assert!(
                (sum_gpu[i] - sum_cpu[i]).abs() < 1e-6,
                "sum idx {} mismatch: gpu={}, cpu={}",
                i,
                sum_gpu[i],
                sum_cpu[i]
            );
            assert!(
                (norm_gpu[i] - norm_cpu[i]).abs() < 1e-5,
                "norm idx {} mismatch: gpu={}, cpu={}",
                i,
                norm_gpu[i],
                norm_cpu[i]
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
    fn rope_qk_fused_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let pos = 11usize;
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
        ctx.apply_rope_qk(&mut q_gpu, &mut k_gpu, pos, head_dim, freq_base)
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

    #[test]
    fn attention_head_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let layer_idx = 1usize;
        let kv_dim = 4usize;
        let head_dim = 2usize;
        let seq_len = 3usize;
        let kv_head_offset = 2usize;
        let scale = (head_dim as f32).sqrt().recip();

        let key_rows = [
            [0.2f32, -0.1f32, 0.3f32, 0.4f32],
            [0.1f32, 0.25f32, -0.2f32, 0.35f32],
            [-0.15f32, 0.05f32, 0.45f32, -0.3f32],
        ];
        let val_rows = [
            [0.7f32, -0.2f32, 0.1f32, 0.9f32],
            [0.2f32, 0.3f32, -0.4f32, 0.6f32],
            [-0.5f32, 0.8f32, 0.55f32, -0.25f32],
        ];
        for pos in 0..seq_len {
            ctx.kv_store(layer_idx, pos, &key_rows[pos], &val_rows[pos], kv_dim)
                .unwrap();
        }

        let q_head = [0.35f32, -0.6f32];
        let gpu = ctx
            .attention_head(&q_head, layer_idx, seq_len, kv_dim, kv_head_offset, scale)
            .unwrap();

        let mut scores = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let k = key_rows[t];
            let dot = q_head[0] * k[kv_head_offset] + q_head[1] * k[kv_head_offset + 1];
            scores.push(dot * scale);
        }
        tensor::softmax(&mut scores);
        let mut cpu = vec![0.0f32; head_dim];
        for t in 0..seq_len {
            let v = val_rows[t];
            for d in 0..head_dim {
                cpu[d] += scores[t] * v[kv_head_offset + d];
            }
        }

        for i in 0..head_dim {
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
    fn attention_layer_matches_cpu() {
        let ctx = match MetalContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let layer_idx = 2usize;
        let n_heads = 2usize;
        let n_heads_per_kv = 1usize;
        let head_dim = 2usize;
        let kv_dim = 4usize;
        let seq_len = 3usize;
        let scale = (head_dim as f32).sqrt().recip();

        let key_rows = [
            [0.2f32, -0.1f32, 0.3f32, 0.4f32],
            [0.1f32, 0.25f32, -0.2f32, 0.35f32],
            [-0.15f32, 0.05f32, 0.45f32, -0.3f32],
        ];
        let val_rows = [
            [0.7f32, -0.2f32, 0.1f32, 0.9f32],
            [0.2f32, 0.3f32, -0.4f32, 0.6f32],
            [-0.5f32, 0.8f32, 0.55f32, -0.25f32],
        ];
        for pos in 0..seq_len {
            ctx.kv_store(layer_idx, pos, &key_rows[pos], &val_rows[pos], kv_dim)
                .unwrap();
        }

        let q = [0.35f32, -0.6f32, -0.2f32, 0.5f32];
        let gpu = ctx
            .attention_layer(
                &q,
                layer_idx,
                seq_len,
                n_heads,
                n_heads_per_kv,
                head_dim,
                kv_dim,
                scale,
            )
            .unwrap();

        let mut cpu = vec![0.0f32; n_heads * head_dim];
        for qh in 0..n_heads {
            let kv_head = qh / n_heads_per_kv;
            let q_off = qh * head_dim;
            let kv_off = kv_head * head_dim;
            let mut scores = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let k = key_rows[t];
                let dot = q[q_off] * k[kv_off] + q[q_off + 1] * k[kv_off + 1];
                scores.push(dot * scale);
            }
            tensor::softmax(&mut scores);
            for t in 0..seq_len {
                let v = val_rows[t];
                cpu[q_off] += scores[t] * v[kv_off];
                cpu[q_off + 1] += scores[t] * v[kv_off + 1];
            }
        }

        for i in 0..cpu.len() {
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
