#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipeline_f32 = nil;
static id<MTLComputePipelineState> g_pipeline_f16 = nil;
static id<MTLComputePipelineState> g_pipeline_q4_0 = nil;
static id<MTLComputePipelineState> g_pipeline_rms_norm = nil;
static id<MTLComputePipelineState> g_pipeline_apply_rope = nil;
static id<MTLComputePipelineState> g_pipeline_softmax = nil;
static dispatch_once_t g_init_once;
static char *g_init_error = NULL;

static void set_error(char **err, const char *msg) {
    if (err == NULL) {
        return;
    }
    if (msg == NULL) {
        *err = NULL;
        return;
    }
    *err = strdup(msg);
}

static void set_init_error(const char *msg) {
    if (g_init_error != NULL) {
        free(g_init_error);
    }
    g_init_error = strdup(msg ? msg : "unknown Metal init error");
}

static const char *shader_source =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matvec_f32(\n"
    "    const device float* matrix [[buffer(0)]],\n"
    "    const device float* vec    [[buffer(1)]],\n"
    "    device float* out          [[buffer(2)]],\n"
    "    constant uint& rows        [[buffer(3)]],\n"
    "    constant uint& cols        [[buffer(4)]],\n"
    "    uint gid                   [[thread_position_in_grid]]) {\n"
    "    if (gid >= rows) return;\n"
    "    uint base = gid * cols;\n"
    "    float sum = 0.0f;\n"
    "    for (uint c = 0; c < cols; ++c) {\n"
    "        sum += matrix[base + c] * vec[c];\n"
    "    }\n"
    "    out[gid] = sum;\n"
    "}\n"
    "kernel void matvec_f16(\n"
    "    const device half* matrix  [[buffer(0)]],\n"
    "    const device float* vec    [[buffer(1)]],\n"
    "    device float* out          [[buffer(2)]],\n"
    "    constant uint& rows        [[buffer(3)]],\n"
    "    constant uint& cols        [[buffer(4)]],\n"
    "    uint gid                   [[thread_position_in_grid]]) {\n"
    "    if (gid >= rows) return;\n"
    "    uint base = gid * cols;\n"
    "    float sum = 0.0f;\n"
    "    for (uint c = 0; c < cols; ++c) {\n"
    "        sum += float(matrix[base + c]) * vec[c];\n"
    "    }\n"
    "    out[gid] = sum;\n"
    "}\n"
    "kernel void matvec_q4_0(\n"
    "    const device uchar* matrix [[buffer(0)]],\n"
    "    const device float* vec    [[buffer(1)]],\n"
    "    device float* out          [[buffer(2)]],\n"
    "    constant uint& rows        [[buffer(3)]],\n"
    "    constant uint& cols        [[buffer(4)]],\n"
    "    uint gid                   [[thread_position_in_grid]]) {\n"
    "    if (gid >= rows) return;\n"
    "    uint row_bytes = (cols / 32u) * 18u;\n"
    "    uint base = gid * row_bytes;\n"
    "    float sum = 0.0f;\n"
    "    uint n_blocks = cols / 32u;\n"
    "    for (uint b = 0; b < n_blocks; ++b) {\n"
    "        uint bo = base + b * 18u;\n"
    "        ushort scale_bits = ushort(matrix[bo]) | (ushort(matrix[bo + 1]) << 8);\n"
    "        float scale = float(as_type<half>(scale_bits));\n"
    "        uint vi = b * 32u;\n"
    "        float block_sum = 0.0f;\n"
    "        for (uint i = 0; i < 16u; ++i) {\n"
    "            uchar packed = matrix[bo + 2u + i];\n"
    "            int lo = int(packed & 0x0Fu) - 8;\n"
    "            int hi = int((packed >> 4) & 0x0Fu) - 8;\n"
    "            block_sum += float(lo) * vec[vi + i];\n"
    "            block_sum += float(hi) * vec[vi + 16u + i];\n"
    "        }\n"
    "        sum += scale * block_sum;\n"
    "    }\n"
    "    out[gid] = sum;\n"
    "}\n"
    "kernel void rms_norm_f32(\n"
    "    const device float* x      [[buffer(0)]],\n"
    "    const device float* weight [[buffer(1)]],\n"
    "    device float* out          [[buffer(2)]],\n"
    "    constant uint& n           [[buffer(3)]],\n"
    "    constant float& eps        [[buffer(4)]],\n"
    "    uint gid                   [[thread_position_in_grid]]) {\n"
    "    if (gid >= n) return;\n"
    "    float ss = 0.0f;\n"
    "    for (uint i = 0; i < n; ++i) {\n"
    "        ss += x[i] * x[i];\n"
    "    }\n"
    "    float rms = sqrt(ss / float(n) + eps);\n"
    "    out[gid] = (x[gid] / rms) * weight[gid];\n"
    "}\n"
    "kernel void apply_rope_f32(\n"
    "    const device float* input [[buffer(0)]],\n"
    "    device float* out         [[buffer(1)]],\n"
    "    constant uint& len        [[buffer(2)]],\n"
    "    constant uint& pos        [[buffer(3)]],\n"
    "    constant uint& head_dim   [[buffer(4)]],\n"
    "    constant float& freq_base [[buffer(5)]],\n"
    "    uint gid                  [[thread_position_in_grid]]) {\n"
    "    uint n_pairs = len / 2u;\n"
    "    if (gid >= n_pairs) return;\n"
    "    uint idx = gid * 2u;\n"
    "    uint i = idx % head_dim;\n"
    "    float freq = pow(freq_base, -float(i) / float(head_dim));\n"
    "    float theta = float(pos) * freq;\n"
    "    float c = cos(theta);\n"
    "    float s = sin(theta);\n"
    "    float x0 = input[idx];\n"
    "    float x1 = input[idx + 1u];\n"
    "    out[idx] = x0 * c - x1 * s;\n"
    "    out[idx + 1u] = x0 * s + x1 * c;\n"
    "}\n"
    "kernel void softmax_f32(\n"
    "    const device float* input [[buffer(0)]],\n"
    "    device float* out         [[buffer(1)]],\n"
    "    constant uint& n          [[buffer(2)]],\n"
    "    uint gid                  [[thread_position_in_grid]]) {\n"
    "    if (gid > 0u) return;\n"
    "    if (n == 0u) return;\n"
    "    float max_v = input[0];\n"
    "    for (uint i = 1u; i < n; ++i) {\n"
    "        if (input[i] > max_v) max_v = input[i];\n"
    "    }\n"
    "    float sum = 0.0f;\n"
    "    for (uint i = 0u; i < n; ++i) {\n"
    "        float v = exp(input[i] - max_v);\n"
    "        out[i] = v;\n"
    "        sum += v;\n"
    "    }\n"
    "    if (sum == 0.0f) return;\n"
    "    for (uint i = 0u; i < n; ++i) {\n"
    "        out[i] = out[i] / sum;\n"
    "    }\n"
    "}\n";

static void init_metal_once(void) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device == nil) {
            set_init_error("MTLCreateSystemDefaultDevice returned nil");
            return;
        }
        g_queue = [g_device newCommandQueue];
        if (g_queue == nil) {
            set_init_error("failed to create Metal command queue");
            return;
        }

        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [g_device newLibraryWithSource:[NSString stringWithUTF8String:shader_source]
                                                   options:opts
                                                     error:&err];
        if (lib == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }

        id<MTLFunction> f32_fn = [lib newFunctionWithName:@"matvec_f32"];
        id<MTLFunction> f16_fn = [lib newFunctionWithName:@"matvec_f16"];
        id<MTLFunction> q4_fn = [lib newFunctionWithName:@"matvec_q4_0"];
        id<MTLFunction> rms_fn = [lib newFunctionWithName:@"rms_norm_f32"];
        id<MTLFunction> rope_fn = [lib newFunctionWithName:@"apply_rope_f32"];
        id<MTLFunction> softmax_fn = [lib newFunctionWithName:@"softmax_f32"];
        if (f32_fn == nil || f16_fn == nil || q4_fn == nil || rms_fn == nil || rope_fn == nil || softmax_fn == nil) {
            set_init_error("failed to load matvec functions from Metal library");
            return;
        }

        g_pipeline_f32 = [g_device newComputePipelineStateWithFunction:f32_fn error:&err];
        if (g_pipeline_f32 == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }
        g_pipeline_f16 = [g_device newComputePipelineStateWithFunction:f16_fn error:&err];
        if (g_pipeline_f16 == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }
        g_pipeline_q4_0 = [g_device newComputePipelineStateWithFunction:q4_fn error:&err];
        if (g_pipeline_q4_0 == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }
        g_pipeline_rms_norm = [g_device newComputePipelineStateWithFunction:rms_fn error:&err];
        if (g_pipeline_rms_norm == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }
        g_pipeline_apply_rope = [g_device newComputePipelineStateWithFunction:rope_fn error:&err];
        if (g_pipeline_apply_rope == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }
        g_pipeline_softmax = [g_device newComputePipelineStateWithFunction:softmax_fn error:&err];
        if (g_pipeline_softmax == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }
    }
}

static int run_matvec(
    id<MTLComputePipelineState> pipeline,
    const void *matrix,
    size_t matrix_bytes,
    const float *vec,
    uint32_t rows,
    uint32_t cols,
    float *out,
    char **err
) {
    if (matrix == NULL || vec == NULL || out == NULL) {
        set_error(err, "null pointer input for matvec");
        return 1;
    }
    if (rows == 0 || cols == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> matrix_buf = [g_device newBufferWithBytes:matrix
                                                          length:matrix_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> vec_buf = [g_device newBufferWithBytes:vec
                                                      length:(NSUInteger)cols * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [g_device newBufferWithLength:(NSUInteger)rows * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (matrix_buf == nil || vec_buf == nil || out_buf == nil) {
            set_error(err, "failed to allocate Metal buffers");
            return 1;
        }

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (cb == nil) {
            set_error(err, "failed to create command buffer");
            return 1;
        }

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            set_error(err, "failed to create compute command encoder");
            return 1;
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:matrix_buf offset:0 atIndex:0];
        [enc setBuffer:vec_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&cols length:sizeof(uint32_t) atIndex:4];

        NSUInteger thread_width = pipeline.threadExecutionWidth;
        if (thread_width == 0) {
            thread_width = 1;
        }
        NSUInteger tg_width = rows < thread_width ? rows : thread_width;
        if (tg_width == 0) {
            tg_width = 1;
        }
        MTLSize threads_per_tg = MTLSizeMake(tg_width, 1, 1);
        MTLSize tg_count = MTLSizeMake(((NSUInteger)rows + tg_width - 1) / tg_width, 1, 1);
        [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:threads_per_tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            const char *msg = cb.error.localizedDescription.UTF8String;
            set_error(err, msg ? msg : "Metal command buffer failed");
            return 1;
        }

        memcpy(out, out_buf.contents, (NSUInteger)rows * sizeof(float));
        return 0;
    }
}

static int run_rms_norm(
    const float *x,
    const float *weight,
    float *out,
    uint32_t n,
    float eps,
    char **err
) {
    if (x == NULL || weight == NULL || out == NULL) {
        set_error(err, "null pointer input for rms_norm");
        return 1;
    }
    if (n == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x
                                                    length:(NSUInteger)n * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = [g_device newBufferWithBytes:weight
                                                    length:(NSUInteger)n * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [g_device newBufferWithLength:(NSUInteger)n * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (x_buf == nil || w_buf == nil || out_buf == nil) {
            set_error(err, "failed to allocate Metal buffers for rms_norm");
            return 1;
        }

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (cb == nil) {
            set_error(err, "failed to create command buffer for rms_norm");
            return 1;
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            set_error(err, "failed to create encoder for rms_norm");
            return 1;
        }

        [enc setComputePipelineState:g_pipeline_rms_norm];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:w_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];

        NSUInteger thread_width = g_pipeline_rms_norm.threadExecutionWidth;
        if (thread_width == 0) thread_width = 1;
        NSUInteger tg_width = n < thread_width ? n : thread_width;
        if (tg_width == 0) tg_width = 1;
        MTLSize threads_per_tg = MTLSizeMake(tg_width, 1, 1);
        MTLSize tg_count = MTLSizeMake(((NSUInteger)n + tg_width - 1) / tg_width, 1, 1);
        [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:threads_per_tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            const char *msg = cb.error.localizedDescription.UTF8String;
            set_error(err, msg ? msg : "Metal rms_norm command failed");
            return 1;
        }

        memcpy(out, out_buf.contents, (NSUInteger)n * sizeof(float));
        return 0;
    }
}

static int run_apply_rope(
    const float *input,
    float *out,
    uint32_t len,
    uint32_t pos,
    uint32_t head_dim,
    float freq_base,
    char **err
) {
    if (input == NULL || out == NULL) {
        set_error(err, "null pointer input for apply_rope");
        return 1;
    }
    if (len == 0) {
        return 0;
    }
    if (head_dim == 0 || (head_dim % 2u) != 0u) {
        set_error(err, "apply_rope requires non-zero even head_dim");
        return 1;
    }
    if ((len % head_dim) != 0u) {
        set_error(err, "apply_rope requires len % head_dim == 0");
        return 1;
    }

    @autoreleasepool {
        id<MTLBuffer> in_buf = [g_device newBufferWithBytes:input
                                                     length:(NSUInteger)len * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [g_device newBufferWithLength:(NSUInteger)len * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (in_buf == nil || out_buf == nil) {
            set_error(err, "failed to allocate Metal buffers for apply_rope");
            return 1;
        }

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (cb == nil) {
            set_error(err, "failed to create command buffer for apply_rope");
            return 1;
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            set_error(err, "failed to create encoder for apply_rope");
            return 1;
        }

        [enc setComputePipelineState:g_pipeline_apply_rope];
        [enc setBuffer:in_buf offset:0 atIndex:0];
        [enc setBuffer:out_buf offset:0 atIndex:1];
        [enc setBytes:&len length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&pos length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&freq_base length:sizeof(float) atIndex:5];

        uint32_t n_pairs = len / 2u;
        NSUInteger thread_width = g_pipeline_apply_rope.threadExecutionWidth;
        if (thread_width == 0) thread_width = 1;
        NSUInteger tg_width = n_pairs < thread_width ? n_pairs : thread_width;
        if (tg_width == 0) tg_width = 1;
        MTLSize threads_per_tg = MTLSizeMake(tg_width, 1, 1);
        MTLSize tg_count = MTLSizeMake(((NSUInteger)n_pairs + tg_width - 1) / tg_width, 1, 1);
        [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:threads_per_tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            const char *msg = cb.error.localizedDescription.UTF8String;
            set_error(err, msg ? msg : "Metal apply_rope command failed");
            return 1;
        }

        memcpy(out, out_buf.contents, (NSUInteger)len * sizeof(float));
        return 0;
    }
}

static int run_softmax(
    const float *input,
    float *out,
    uint32_t n,
    char **err
) {
    if (input == NULL || out == NULL) {
        set_error(err, "null pointer input for softmax");
        return 1;
    }
    if (n == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> in_buf = [g_device newBufferWithBytes:input
                                                     length:(NSUInteger)n * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [g_device newBufferWithLength:(NSUInteger)n * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (in_buf == nil || out_buf == nil) {
            set_error(err, "failed to allocate Metal buffers for softmax");
            return 1;
        }

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (cb == nil) {
            set_error(err, "failed to create command buffer for softmax");
            return 1;
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            set_error(err, "failed to create encoder for softmax");
            return 1;
        }

        [enc setComputePipelineState:g_pipeline_softmax];
        [enc setBuffer:in_buf offset:0 atIndex:0];
        [enc setBuffer:out_buf offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];

        MTLSize threads_per_tg = MTLSizeMake(1, 1, 1);
        MTLSize tg_count = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:threads_per_tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            const char *msg = cb.error.localizedDescription.UTF8String;
            set_error(err, msg ? msg : "Metal softmax command failed");
            return 1;
        }

        memcpy(out, out_buf.contents, (NSUInteger)n * sizeof(float));
        return 0;
    }
}

int rsllm_metal_matvec_f32(
    const float *matrix,
    const float *vec,
    float *out,
    uint32_t rows,
    uint32_t cols,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    size_t matrix_bytes = (size_t)rows * (size_t)cols * sizeof(float);
    return run_matvec(g_pipeline_f32, matrix, matrix_bytes, vec, rows, cols, out, err);
}

int rsllm_metal_is_available(char **err) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 0;
    }
    return 1;
}

int rsllm_metal_matvec_f16(
    const uint16_t *matrix,
    const float *vec,
    float *out,
    uint32_t rows,
    uint32_t cols,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    size_t matrix_bytes = (size_t)rows * (size_t)cols * sizeof(uint16_t);
    return run_matvec(g_pipeline_f16, matrix, matrix_bytes, vec, rows, cols, out, err);
}

int rsllm_metal_matvec_q4_0(
    const uint8_t *matrix,
    const float *vec,
    float *out,
    uint32_t rows,
    uint32_t cols,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    if (cols % 32u != 0u) {
        set_error(err, "Q4_0 matvec requires cols % 32 == 0");
        return 1;
    }
    size_t row_bytes = ((size_t)cols / 32u) * 18u;
    size_t matrix_bytes = (size_t)rows * row_bytes;
    return run_matvec(g_pipeline_q4_0, matrix, matrix_bytes, vec, rows, cols, out, err);
}

int rsllm_metal_rms_norm(
    const float *x,
    const float *weight,
    float *out,
    uint32_t n,
    float eps,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    return run_rms_norm(x, weight, out, n, eps, err);
}

int rsllm_metal_apply_rope(
    const float *input,
    float *out,
    uint32_t len,
    uint32_t pos,
    uint32_t head_dim,
    float freq_base,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    return run_apply_rope(input, out, len, pos, head_dim, freq_base, err);
}

int rsllm_metal_softmax(
    const float *input,
    float *out,
    uint32_t n,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    return run_softmax(input, out, n, err);
}

void rsllm_metal_free_error(char *err) {
    if (err != NULL) {
        free(err);
    }
}
