# 第 7 章：后端抽象与 GPU 加速

> **学习目标**：理解 Backend trait 的设计，掌握 CPU/Metal 后端的实现，理解层级卸载和融合算子。
>
> **对应源码**：`src/backend.rs`（552 行）、`src/metal/`（context.rs, buffer.rs, weights.rs, ffi/）
>
> **预计时间**：4-5 小时

---

## 7.1 为什么需要后端抽象？

模型的前向传播涉及大量矩阵运算。在 CPU 上运行可以工作，但速度慢。GPU 可以大幅加速，但不是所有环境都有 GPU。

**解决方案**：用 Rust 的 trait 定义统一接口，不同后端各自实现：

```
┌─────────────────────────────────┐
│         LlamaModel              │
│   forward() 调用 Backend 方法    │
└──────────┬──────────────────────┘
           │ dyn Backend
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌──────────┐
│CpuBackend│  │MetalBackend│
│ (默认)   │  │ (Apple GPU)│
└────────┘  └──────────┘
```

---

## 7.2 Backend Trait

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    // 核心算子
    fn matmul_vec(&self, mat: &TensorRef, vec: &[f32]) -> Vec<f32>;
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32>;
    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freq_base: f32);
    fn softmax(&self, x: &mut [f32]);
    fn silu(&self, x: &[f32]) -> Vec<f32>;
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32>;

    // 融合算子（默认实现）
    fn rope_qk(&self, q: &mut [f32], k: &mut [f32], ...) {
        self.rope(q, k, pos, head_dim, freq_base);  // 默认调用 rope
    }
    fn add_rms_norm(&self, a: &[f32], b: &[f32], weight: &[f32], eps: f32)
        -> (Vec<f32>, Vec<f32>) {
        let sum = self.add(a, b);
        let norm = self.rms_norm(&sum, weight, eps);
        (sum, norm)  // 默认分两步
    }

    // 注意力相关
    fn store_kv(&self, ...) {}  // 默认空操作
    fn attention_head(&self, ...) -> Vec<f32> { /* CPU 实现 */ }
    fn attention_layer(&self, ...) -> Vec<f32> { /* 循环调用 attention_head */ }

    // 层级卸载决策
    fn should_use_layer_gpu(&self, _layer_idx: usize, _n_layers: usize) -> bool {
        false  // 默认不使用 GPU
    }
}
```

### 设计亮点

1. **`Send + Sync`**：确保后端可以安全地跨线程使用
2. **默认实现**：融合算子有默认的"分步"实现，GPU 后端可以覆盖为更高效的融合版本
3. **渐进式 GPU**：`should_use_layer_gpu()` 让每一层独立决定是否使用 GPU

---

## 7.3 CPU 后端

CPU 后端是最简单的实现——直接委托给 `tensor` 模块的函数：

```rust
#[derive(Debug, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &'static str { "cpu" }

    fn matmul_vec(&self, mat: &TensorRef, vec: &[f32]) -> Vec<f32> {
        tensor::matmul_vec(mat, vec)  // 直接调用 tensor 模块
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        tensor::rms_norm(x, weight, eps)
    }

    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
        tensor::rope(q, k, pos, head_dim, freq_base);
    }

    fn softmax(&self, x: &mut [f32]) { tensor::softmax(x); }
    fn silu(&self, x: &[f32]) -> Vec<f32> { tensor::silu(x) }
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> { tensor::add(a, b) }
    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> { tensor::mul(a, b) }
}
```

CPU 后端是**零开销抽象**——trait 方法直接内联到 tensor 函数调用。

---

## 7.4 Metal 后端（Apple GPU）

### 条件编译

Metal 后端通过 Cargo feature 控制：

```toml
# Cargo.toml
[features]
metal = []
```

```rust
#[cfg(feature = "metal")]
pub struct MetalBackend {
    cpu_fallback: CpuBackend,    // CPU 回退
    context: MetalContext,        // Metal GPU 上下文
    weights: MetalWeightStore,   // 权重缓存
    gpu_layers: usize,           // GPU 层数
}
```

### 回退策略

Metal 后端的每个方法都遵循相同模式：

```rust
fn matmul_vec(&self, mat: &TensorRef, vec: &[f32]) -> Vec<f32> {
    // 1. 如果 GPU 层数为 0，直接用 CPU
    if self.gpu_layers == 0 {
        return self.cpu_fallback.matmul_vec(mat, vec);
    }

    // 2. 注册权重到 GPU 缓存
    self.weights.register_tensor(&mat.info, mat.data);

    // 3. 根据数据类型选择 GPU kernel
    let gpu_result = match mat.info.dtype {
        GgufDType::F32  => self.context.matvec_f32(mat.data, rows, cols, vec),
        GgufDType::F16  => self.context.matvec_f16(mat.data, rows, cols, vec),
        GgufDType::Q4_0 => self.context.matvec_q4_0(mat.data, rows, cols, vec),
        _ => return self.cpu_fallback.matmul_vec(mat, vec),  // 不支持的类型回退
    };

    // 4. GPU 失败时回退到 CPU
    match gpu_result {
        Ok(out) => out,
        Err(_) => self.cpu_fallback.matmul_vec(mat, vec),
    }
}
```

**三层防御**：
1. `gpu_layers == 0` → 完全跳过 GPU
2. 不支持的数据类型 → 回退 CPU
3. GPU 执行失败 → 回退 CPU

这保证了 Metal 后端永远不会崩溃——最坏情况下退化为 CPU 执行。

---

## 7.5 层级卸载

不是所有层都需要在 GPU 上运行。`--gpu-layers N` 参数控制将多少层卸载到 GPU：

```rust
pub fn should_offload_layer(gpu_layers: usize, layer_idx: usize, n_layers: usize) -> bool {
    if gpu_layers == 0 {
        return false;
    }
    let start = n_layers.saturating_sub(gpu_layers);
    layer_idx >= start && layer_idx < n_layers
}
```

**策略**：从最后一层开始卸载（后面的层对输出影响更直接）：

```
22 层模型，gpu_layers = 8:

层 0-13:  CPU  ████████████████
层 14-21: GPU  ▓▓▓▓▓▓▓▓

start = 22 - 8 = 14
layer_idx >= 14 → GPU
```

**为什么从后往前？**
- 后面的层更接近输出，加速效果更明显
- 前面的层可以在 CPU 上并行准备数据
- 适应不同显存大小——显存小就少卸载几层

---

## 7.6 融合算子

### rope_qk：融合 RoPE

默认实现分别对 Q 和 K 调用 `rope()`。Metal 后端可以用一次 GPU 调度同时处理：

```rust
// 默认（两次调用）
fn rope_qk(&self, q, k, pos, head_dim, freq_base) {
    self.rope(q, k, pos, head_dim, freq_base);
}

// Metal 融合版本（一次 GPU 调度）
fn rope_qk(&self, q, k, pos, head_dim, freq_base) {
    self.context.apply_rope_qk(q, k, pos, head_dim, freq_base);
}
```

### add_rms_norm：融合加法和归一化

```rust
// 默认（两步）
fn add_rms_norm(&self, a, b, weight, eps) -> (Vec<f32>, Vec<f32>) {
    let sum = self.add(a, b);
    let norm = self.rms_norm(&sum, weight, eps);
    (sum, norm)
}

// Metal 融合版本（一次 GPU 调度）
fn add_rms_norm(&self, a, b, weight, eps) -> (Vec<f32>, Vec<f32>) {
    self.context.add_rms_norm(a, b, weight, eps)
}
```

**融合的好处**：
- 减少 GPU 调度次数（每次调度有固定开销）
- 减少中间结果的内存读写
- 对于小向量操作，调度开销可能比计算本身还大

---

## 7.7 Metal FFI 桥接

Rust 通过 FFI（Foreign Function Interface）调用 Objective-C 编写的 Metal 代码：

```
Rust 代码                    Objective-C 桥接              Metal GPU
─────────                    ──────────────              ─────────
MetalContext                 metal_bridge.m              matvec.metal
  .matvec_f32() ──FFI──→ rsllm_metal_matvec_f32() ──→ GPU kernel
  .rms_norm()   ──FFI──→ rsllm_metal_rms_norm()   ──→ GPU kernel
  .softmax()    ──FFI──→ rsllm_metal_softmax()     ──→ GPU kernel
```

### 构建流程

`build.rs` 负责编译 Objective-C 桥接代码：

```rust
// build.rs
fn main() {
    // 只在启用 metal feature 且目标是 macOS 时编译
    if !metal_enabled || target_os != "macos" { return; }

    // 1. 用 xcrun clang 编译 Objective-C
    Command::new("xcrun").args(["clang", "-fobjc-arc", "-O2", "-c",
        "src/metal/ffi/metal_bridge.m", "-o", obj_path]);

    // 2. 打包为静态库
    Command::new("ar").args(["rcs", lib_path, obj_path]);

    // 3. 链接 Metal 和 Foundation 框架
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
```

### 错误处理

所有 FFI 函数返回状态码 + 错误指针：

```rust
// C 函数签名
fn rsllm_metal_matvec_f32(
    matrix: *const f32, vec: *const f32, out: *mut f32,
    rows: u32, cols: u32,
    err: *mut *mut c_char,  // 错误消息输出
) -> i32;                    // 0 = 成功，非 0 = 失败

// Rust 端处理
let mut err_ptr: *mut c_char = std::ptr::null_mut();
let status = unsafe { rsllm_metal_matvec_f32(..., &mut err_ptr) };
if status == 0 {
    Ok(out)
} else {
    Err(take_error_message(err_ptr))  // 读取并释放错误消息
}
```

---

## 7.8 权重缓存

`MetalWeightStore` 跟踪已上传到 GPU 的权重，避免重复传输：

```rust
pub struct MetalWeightStore {
    inner: Mutex<Inner>,  // 线程安全
}

struct Inner {
    records: HashMap<String, WeightRecord>,  // 按名称索引
    uploaded_tensors: usize,
    uploaded_bytes: usize,
}
```

通过指纹（fingerprint）检测权重是否已上传：

```rust
pub fn register_tensor(&self, info: &TensorInfo, data: &[u8]) -> bool {
    let fp = fingerprint(data);  // 哈希前 128 + 后 128 字节
    // 如果名称、指纹、大小、类型都匹配 → 缓存命中
    // 否则 → 新上传
}
```

---

## 7.9 后端工厂

`build_backend()` 根据用户选择创建后端：

```rust
pub fn build_backend(device: DeviceKind, gpu_layers: usize) -> Result<Box<dyn Backend>, String> {
    match device {
        DeviceKind::Cpu => Ok(Box::new(CpuBackend::new())),
        DeviceKind::Metal => {
            #[cfg(feature = "metal")]
            { Ok(Box::new(MetalBackend::new(gpu_layers)?)) }
            #[cfg(not(feature = "metal"))]
            { Err("Metal backend is disabled. Rebuild with `--features metal`.".to_string()) }
        }
    }
}
```

返回 `Box<dyn Backend>`——动态分发，运行时选择后端。

---

## 7.10 小结

✅ 本章你学到了：

- [ ] Backend trait：统一的算子接口，支持多后端
- [ ] CpuBackend：直接委托给 tensor 模块，零开销
- [ ] MetalBackend：GPU 加速 + 三层回退保护
- [ ] 层级卸载：从后往前卸载，适应不同显存
- [ ] 融合算子：减少 GPU 调度开销
- [ ] FFI 桥接：Rust → Objective-C → Metal GPU
- [ ] 权重缓存：避免重复上传

### 关键设计决策

1. **为什么用 `dyn Backend` 而不是泛型？** 后端在运行时由命令行参数决定，需要动态分发。泛型需要编译时确定类型。
2. **为什么 Metal 后端总是包含 CPU 回退？** GPU 不支持所有量化格式（如 Q6_K），回退保证功能完整。
3. **为什么用条件编译而不是运行时检测？** Metal 依赖 macOS 框架，在 Linux 上编译会失败。条件编译确保跨平台兼容。

**下一章**：[第 8 章：完整推理流程](./08-inference-pipeline.md) —— 将所有模块串联，理解从命令行到文本输出的完整流程。