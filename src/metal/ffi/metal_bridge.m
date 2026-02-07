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
static id<MTLComputePipelineState> g_pipeline_attn_head = nil;
static id<MTLComputePipelineState> g_pipeline_attn_layer = nil;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_matrix_cache = nil;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *g_key_cache = nil;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *g_val_cache = nil;
static id<MTLBuffer> g_matvec_vec_buf = nil;
static id<MTLBuffer> g_matvec_out_buf = nil;
static id<MTLBuffer> g_attn_q_buf = nil;
static id<MTLBuffer> g_attn_out_buf = nil;
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
    "}\n"
    "kernel void attn_head_f32(\n"
    "    const device float* q         [[buffer(0)]],\n"
    "    const device float* key_cache [[buffer(1)]],\n"
    "    const device float* val_cache [[buffer(2)]],\n"
    "    device float* out             [[buffer(3)]],\n"
    "    constant uint& seq_len        [[buffer(4)]],\n"
    "    constant uint& kv_dim         [[buffer(5)]],\n"
    "    constant uint& kv_head_offset [[buffer(6)]],\n"
    "    constant uint& head_dim       [[buffer(7)]],\n"
    "    constant float& scale         [[buffer(8)]],\n"
    "    uint gid                      [[thread_position_in_grid]]) {\n"
    "    if (gid >= head_dim) return;\n"
    "    if (seq_len == 0u) {\n"
    "        out[gid] = 0.0f;\n"
    "        return;\n"
    "    }\n"
    "    float max_score = -INFINITY;\n"
    "    for (uint t = 0u; t < seq_len; ++t) {\n"
    "        uint base = t * kv_dim + kv_head_offset;\n"
    "        float dot = 0.0f;\n"
    "        for (uint d = 0u; d < head_dim; ++d) {\n"
    "            dot += q[d] * key_cache[base + d];\n"
    "        }\n"
    "        float s = dot * scale;\n"
    "        if (s > max_score) max_score = s;\n"
    "    }\n"
    "    float denom = 0.0f;\n"
    "    for (uint t = 0u; t < seq_len; ++t) {\n"
    "        uint base = t * kv_dim + kv_head_offset;\n"
    "        float dot = 0.0f;\n"
    "        for (uint d = 0u; d < head_dim; ++d) {\n"
    "            dot += q[d] * key_cache[base + d];\n"
    "        }\n"
    "        denom += exp(dot * scale - max_score);\n"
    "    }\n"
    "    if (denom <= 0.0f) {\n"
    "        out[gid] = 0.0f;\n"
    "        return;\n"
    "    }\n"
    "    float acc = 0.0f;\n"
    "    for (uint t = 0u; t < seq_len; ++t) {\n"
    "        uint base = t * kv_dim + kv_head_offset;\n"
    "        float dot = 0.0f;\n"
    "        for (uint d = 0u; d < head_dim; ++d) {\n"
    "            dot += q[d] * key_cache[base + d];\n"
    "        }\n"
    "        float w = exp(dot * scale - max_score) / denom;\n"
    "        acc += w * val_cache[base + gid];\n"
    "    }\n"
    "    out[gid] = acc;\n"
    "}\n"
    "kernel void attn_layer_f32(\n"
    "    const device float* q             [[buffer(0)]],\n"
    "    const device float* key_cache     [[buffer(1)]],\n"
    "    const device float* val_cache     [[buffer(2)]],\n"
    "    device float* out                 [[buffer(3)]],\n"
    "    constant uint& seq_len            [[buffer(4)]],\n"
    "    constant uint& n_heads            [[buffer(5)]],\n"
    "    constant uint& n_heads_per_kv     [[buffer(6)]],\n"
    "    constant uint& head_dim           [[buffer(7)]],\n"
    "    constant uint& kv_dim             [[buffer(8)]],\n"
    "    constant float& scale             [[buffer(9)]],\n"
    "    uint gid                          [[thread_position_in_grid]]) {\n"
    "    uint total = n_heads * head_dim;\n"
    "    if (gid >= total) return;\n"
    "    uint qh = gid / head_dim;\n"
    "    uint d_out = gid % head_dim;\n"
    "    uint q_base = qh * head_dim;\n"
    "    uint kv_head = qh / n_heads_per_kv;\n"
    "    uint kv_head_offset = kv_head * head_dim;\n"
    "    if (seq_len == 0u) {\n"
    "        out[gid] = 0.0f;\n"
    "        return;\n"
    "    }\n"
    "    float max_score = -INFINITY;\n"
    "    for (uint t = 0u; t < seq_len; ++t) {\n"
    "        uint base = t * kv_dim + kv_head_offset;\n"
    "        float dot = 0.0f;\n"
    "        for (uint d = 0u; d < head_dim; ++d) {\n"
    "            dot += q[q_base + d] * key_cache[base + d];\n"
    "        }\n"
    "        float s = dot * scale;\n"
    "        if (s > max_score) max_score = s;\n"
    "    }\n"
    "    float denom = 0.0f;\n"
    "    for (uint t = 0u; t < seq_len; ++t) {\n"
    "        uint base = t * kv_dim + kv_head_offset;\n"
    "        float dot = 0.0f;\n"
    "        for (uint d = 0u; d < head_dim; ++d) {\n"
    "            dot += q[q_base + d] * key_cache[base + d];\n"
    "        }\n"
    "        denom += exp(dot * scale - max_score);\n"
    "    }\n"
    "    if (denom <= 0.0f) {\n"
    "        out[gid] = 0.0f;\n"
    "        return;\n"
    "    }\n"
    "    float acc = 0.0f;\n"
    "    for (uint t = 0u; t < seq_len; ++t) {\n"
    "        uint base = t * kv_dim + kv_head_offset;\n"
    "        float dot = 0.0f;\n"
    "        for (uint d = 0u; d < head_dim; ++d) {\n"
    "            dot += q[q_base + d] * key_cache[base + d];\n"
    "        }\n"
    "        float w = exp(dot * scale - max_score) / denom;\n"
    "        acc += w * val_cache[base + d_out];\n"
    "    }\n"
    "    out[gid] = acc;\n"
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
        id<MTLFunction> attn_fn = [lib newFunctionWithName:@"attn_head_f32"];
        id<MTLFunction> attn_layer_fn = [lib newFunctionWithName:@"attn_layer_f32"];
        if (f32_fn == nil || f16_fn == nil || q4_fn == nil || rms_fn == nil ||
            rope_fn == nil || softmax_fn == nil || attn_fn == nil || attn_layer_fn == nil) {
            set_init_error("failed to load required Metal functions from library");
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
        g_pipeline_attn_head = [g_device newComputePipelineStateWithFunction:attn_fn error:&err];
        if (g_pipeline_attn_head == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }
        g_pipeline_attn_layer = [g_device newComputePipelineStateWithFunction:attn_layer_fn error:&err];
        if (g_pipeline_attn_layer == nil) {
            set_init_error(err.localizedDescription.UTF8String);
            return;
        }

        g_matrix_cache = [NSMutableDictionary dictionary];
        g_key_cache = [NSMutableDictionary dictionary];
        g_val_cache = [NSMutableDictionary dictionary];
    }
}

static id<MTLBuffer> ensure_scratch_buffer(id<MTLBuffer> __strong *slot, NSUInteger required_bytes, char **err) {
    if (required_bytes == 0) {
        required_bytes = sizeof(float);
    }
    if (*slot != nil && (*slot).length >= required_bytes) {
        return *slot;
    }
    id<MTLBuffer> buf = [g_device newBufferWithLength:required_bytes options:MTLResourceStorageModeShared];
    if (buf == nil) {
        set_error(err, "failed to allocate Metal scratch buffer");
        return nil;
    }
    *slot = buf;
    return buf;
}

static id<MTLBuffer> get_or_create_matrix_buffer(
    const void *matrix,
    size_t matrix_bytes,
    char **err
) {
    if (g_matrix_cache == nil) {
        set_error(err, "Metal matrix cache is not initialized");
        return nil;
    }
    NSString *key = [NSString stringWithFormat:@"%p:%zu", matrix, matrix_bytes];
    id<MTLBuffer> matrix_buf = g_matrix_cache[key];
    if (matrix_buf != nil) {
        return matrix_buf;
    }

    matrix_buf = [g_device newBufferWithBytes:matrix
                                       length:(NSUInteger)matrix_bytes
                                      options:MTLResourceStorageModeShared];
    if (matrix_buf == nil) {
        set_error(err, "failed to allocate Metal matrix buffer");
        return nil;
    }
    g_matrix_cache[key] = matrix_buf;
    return matrix_buf;
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
        id<MTLBuffer> matrix_buf = get_or_create_matrix_buffer(matrix, matrix_bytes, err);
        if (matrix_buf == nil) {
            return 1;
        }
        id<MTLBuffer> vec_buf = ensure_scratch_buffer(&g_matvec_vec_buf, (NSUInteger)cols * sizeof(float), err);
        id<MTLBuffer> out_buf = ensure_scratch_buffer(&g_matvec_out_buf, (NSUInteger)rows * sizeof(float), err);
        if (vec_buf == nil || out_buf == nil) {
            return 1;
        }
        memcpy(vec_buf.contents, vec, (NSUInteger)cols * sizeof(float));

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

static id<MTLBuffer> ensure_layer_buffer(
    NSMutableDictionary<NSNumber *, id<MTLBuffer>> *cache_dict,
    uint32_t layer,
    NSUInteger required_bytes,
    char **err
) {
    if (cache_dict == nil) {
        set_error(err, "Metal KV cache dictionary is not initialized");
        return nil;
    }

    NSNumber *key = @(layer);
    id<MTLBuffer> buf = cache_dict[key];
    if (buf != nil && buf.length >= required_bytes) {
        return buf;
    }

    NSUInteger new_len = required_bytes;
    if (buf != nil && buf.length > new_len) {
        new_len = buf.length;
    }
    if (new_len == 0) {
        new_len = sizeof(float);
    }
    id<MTLBuffer> new_buf = [g_device newBufferWithLength:new_len options:MTLResourceStorageModeShared];
    if (new_buf == nil) {
        set_error(err, "failed to allocate Metal KV cache buffer");
        return nil;
    }
    if (buf != nil) {
        memcpy(new_buf.contents, buf.contents, buf.length);
    }
    cache_dict[key] = new_buf;
    return new_buf;
}

static int run_kv_store(
    uint32_t layer,
    uint32_t pos,
    const float *key,
    const float *val,
    uint32_t kv_dim,
    char **err
) {
    if (key == NULL || val == NULL) {
        set_error(err, "null pointer input for kv_store");
        return 1;
    }
    if (kv_dim == 0) {
        set_error(err, "kv_store requires kv_dim > 0");
        return 1;
    }

    size_t required_elems = ((size_t)pos + 1u) * (size_t)kv_dim;
    NSUInteger required_bytes = (NSUInteger)(required_elems * sizeof(float));
    id<MTLBuffer> key_buf = ensure_layer_buffer(g_key_cache, layer, required_bytes, err);
    if (key_buf == nil) {
        return 1;
    }
    id<MTLBuffer> val_buf = ensure_layer_buffer(g_val_cache, layer, required_bytes, err);
    if (val_buf == nil) {
        return 1;
    }

    size_t byte_offset = (size_t)pos * (size_t)kv_dim * sizeof(float);
    memcpy((uint8_t *)key_buf.contents + byte_offset, key, (size_t)kv_dim * sizeof(float));
    memcpy((uint8_t *)val_buf.contents + byte_offset, val, (size_t)kv_dim * sizeof(float));
    return 0;
}

static int run_attn_head(
    const float *q,
    uint32_t layer,
    uint32_t seq_len,
    uint32_t kv_dim,
    uint32_t kv_head_offset,
    uint32_t head_dim,
    float scale,
    float *out,
    char **err
) {
    if (q == NULL || out == NULL) {
        set_error(err, "null pointer input for attn_head");
        return 1;
    }
    if (head_dim == 0) {
        return 0;
    }
    if (seq_len == 0) {
        memset(out, 0, (size_t)head_dim * sizeof(float));
        return 0;
    }
    if (kv_head_offset + head_dim > kv_dim) {
        set_error(err, "attn_head requires kv_head_offset + head_dim <= kv_dim");
        return 1;
    }

    NSNumber *layer_key = @(layer);
    id<MTLBuffer> key_buf = g_key_cache[layer_key];
    id<MTLBuffer> val_buf = g_val_cache[layer_key];
    if (key_buf == nil || val_buf == nil) {
        set_error(err, "missing KV cache for requested layer");
        return 1;
    }

    size_t needed_elems = (size_t)(seq_len - 1u) * (size_t)kv_dim + (size_t)kv_head_offset + (size_t)head_dim;
    size_t needed_bytes = needed_elems * sizeof(float);
    if ((size_t)key_buf.length < needed_bytes || (size_t)val_buf.length < needed_bytes) {
        set_error(err, "KV cache buffer is smaller than requested attention range");
        return 1;
    }

    @autoreleasepool {
        id<MTLBuffer> q_buf = ensure_scratch_buffer(&g_attn_q_buf, (NSUInteger)head_dim * sizeof(float), err);
        id<MTLBuffer> out_buf = ensure_scratch_buffer(&g_attn_out_buf, (NSUInteger)head_dim * sizeof(float), err);
        if (q_buf == nil || out_buf == nil) {
            set_error(err, "failed to allocate Metal buffers for attn_head");
            return 1;
        }
        memcpy(q_buf.contents, q, (NSUInteger)head_dim * sizeof(float));

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (cb == nil) {
            set_error(err, "failed to create command buffer for attn_head");
            return 1;
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            set_error(err, "failed to create encoder for attn_head");
            return 1;
        }

        [enc setComputePipelineState:g_pipeline_attn_head];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:key_buf offset:0 atIndex:1];
        [enc setBuffer:val_buf offset:0 atIndex:2];
        [enc setBuffer:out_buf offset:0 atIndex:3];
        [enc setBytes:&seq_len length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&kv_dim length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&kv_head_offset length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&scale length:sizeof(float) atIndex:8];

        NSUInteger thread_width = g_pipeline_attn_head.threadExecutionWidth;
        if (thread_width == 0) thread_width = 1;
        NSUInteger tg_width = head_dim < thread_width ? head_dim : thread_width;
        if (tg_width == 0) tg_width = 1;
        MTLSize threads_per_tg = MTLSizeMake(tg_width, 1, 1);
        MTLSize tg_count = MTLSizeMake(((NSUInteger)head_dim + tg_width - 1) / tg_width, 1, 1);
        [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:threads_per_tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            const char *msg = cb.error.localizedDescription.UTF8String;
            set_error(err, msg ? msg : "Metal attn_head command failed");
            return 1;
        }

        memcpy(out, out_buf.contents, (NSUInteger)head_dim * sizeof(float));
        return 0;
    }
}

static int run_attn_layer(
    const float *q,
    uint32_t layer,
    uint32_t seq_len,
    uint32_t n_heads,
    uint32_t n_heads_per_kv,
    uint32_t head_dim,
    uint32_t kv_dim,
    float scale,
    float *out,
    char **err
) {
    if (q == NULL || out == NULL) {
        set_error(err, "null pointer input for attn_layer");
        return 1;
    }
    if (n_heads == 0 || head_dim == 0) {
        return 0;
    }
    if (n_heads_per_kv == 0 || (n_heads % n_heads_per_kv) != 0u) {
        set_error(err, "attn_layer requires valid n_heads_per_kv");
        return 1;
    }

    size_t total = (size_t)n_heads * (size_t)head_dim;
    if (seq_len == 0) {
        memset(out, 0, total * sizeof(float));
        return 0;
    }

    NSNumber *layer_key = @(layer);
    id<MTLBuffer> key_buf = g_key_cache[layer_key];
    id<MTLBuffer> val_buf = g_val_cache[layer_key];
    if (key_buf == nil || val_buf == nil) {
        set_error(err, "missing KV cache for requested layer");
        return 1;
    }

    uint32_t max_kv_head = (n_heads - 1u) / n_heads_per_kv;
    size_t max_offset = (size_t)max_kv_head * (size_t)head_dim;
    size_t needed_elems = (size_t)(seq_len - 1u) * (size_t)kv_dim + max_offset + (size_t)head_dim;
    size_t needed_bytes = needed_elems * sizeof(float);
    if ((size_t)key_buf.length < needed_bytes || (size_t)val_buf.length < needed_bytes) {
        set_error(err, "KV cache buffer is smaller than requested attention range");
        return 1;
    }

    @autoreleasepool {
        id<MTLBuffer> q_buf = ensure_scratch_buffer(&g_attn_q_buf, (NSUInteger)(total * sizeof(float)), err);
        id<MTLBuffer> out_buf = ensure_scratch_buffer(&g_attn_out_buf, (NSUInteger)(total * sizeof(float)), err);
        if (q_buf == nil || out_buf == nil) {
            set_error(err, "failed to allocate Metal buffers for attn_layer");
            return 1;
        }
        memcpy(q_buf.contents, q, (NSUInteger)(total * sizeof(float)));

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (cb == nil) {
            set_error(err, "failed to create command buffer for attn_layer");
            return 1;
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            set_error(err, "failed to create encoder for attn_layer");
            return 1;
        }

        [enc setComputePipelineState:g_pipeline_attn_layer];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:key_buf offset:0 atIndex:1];
        [enc setBuffer:val_buf offset:0 atIndex:2];
        [enc setBuffer:out_buf offset:0 atIndex:3];
        [enc setBytes:&seq_len length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&n_heads length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&n_heads_per_kv length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&kv_dim length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&scale length:sizeof(float) atIndex:9];

        NSUInteger thread_width = g_pipeline_attn_layer.threadExecutionWidth;
        if (thread_width == 0) thread_width = 1;
        NSUInteger tg_width = (NSUInteger)total < thread_width ? (NSUInteger)total : thread_width;
        if (tg_width == 0) tg_width = 1;
        MTLSize threads_per_tg = MTLSizeMake(tg_width, 1, 1);
        MTLSize tg_count = MTLSizeMake(((NSUInteger)total + tg_width - 1) / tg_width, 1, 1);
        [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:threads_per_tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            const char *msg = cb.error.localizedDescription.UTF8String;
            set_error(err, msg ? msg : "Metal attn_layer command failed");
            return 1;
        }

        memcpy(out, out_buf.contents, (NSUInteger)(total * sizeof(float)));
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

int rsllm_metal_kv_store(
    uint32_t layer,
    uint32_t pos,
    const float *key,
    const float *val,
    uint32_t kv_dim,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    return run_kv_store(layer, pos, key, val, kv_dim, err);
}

int rsllm_metal_attn_head(
    const float *q,
    uint32_t layer,
    uint32_t seq_len,
    uint32_t kv_dim,
    uint32_t kv_head_offset,
    uint32_t head_dim,
    float scale,
    float *out,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    return run_attn_head(
        q,
        layer,
        seq_len,
        kv_dim,
        kv_head_offset,
        head_dim,
        scale,
        out,
        err
    );
}

int rsllm_metal_attn_layer(
    const float *q,
    uint32_t layer,
    uint32_t seq_len,
    uint32_t n_heads,
    uint32_t n_heads_per_kv,
    uint32_t head_dim,
    uint32_t kv_dim,
    float scale,
    float *out,
    char **err
) {
    dispatch_once(&g_init_once, ^{
      init_metal_once();
    });
    if (g_init_error != NULL) {
        set_error(err, g_init_error);
        return 1;
    }
    return run_attn_layer(
        q,
        layer,
        seq_len,
        n_heads,
        n_heads_per_kv,
        head_dim,
        kv_dim,
        scale,
        out,
        err
    );
}

void rsllm_metal_free_error(char *err) {
    if (err != NULL) {
        free(err);
    }
}
