#include <metal_stdlib>
using namespace metal;

kernel void matvec_f32(
    const device float* matrix [[buffer(0)]],
    const device float* vec    [[buffer(1)]],
    device float* out          [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= rows) {
        return;
    }

    uint base = gid * cols;
    float sum = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        sum += matrix[base + c] * vec[c];
    }
    out[gid] = sum;
}

kernel void matvec_f16(
    const device half* matrix  [[buffer(0)]],
    const device float* vec    [[buffer(1)]],
    device float* out          [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= rows) {
        return;
    }

    uint base = gid * cols;
    float sum = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        sum += float(matrix[base + c]) * vec[c];
    }
    out[gid] = sum;
}
