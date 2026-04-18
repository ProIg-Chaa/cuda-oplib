#include "cuda_oplib/rmsnorm.h"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cuda_oplib {
namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static_assert(BLOCK_SIZE % kWarpSize == 0, "BLOCK_SIZE must be a multiple of warpSize");
    __shared__ float shared[BLOCK_SIZE / kWarpSize];

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < (BLOCK_SIZE / kWarpSize)) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void rmsnorm_half2_kernel(
    const half* __restrict__ x,
    const half* __restrict__ gamma,
    half* __restrict__ y,
    int rows,
    int hidden,
    float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const half* row_x = x + row * hidden;
    half* row_y = y + row * hidden;

    const bool aligned =
        ((reinterpret_cast<std::uintptr_t>(row_x) & 0x3) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(row_y) & 0x3) == 0);
    const int vec_hidden = aligned ? (hidden / 2) : 0;
    const int tail_start = vec_hidden * 2;

    const half2* row_x_vec2 = aligned ? reinterpret_cast<const half2*>(row_x) : nullptr;
    half2* row_y_vec2 = aligned ? reinterpret_cast<half2*>(row_y) : nullptr;

    __shared__ float inv_rms;
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < vec_hidden; i += BLOCK_SIZE) {
        const float2 val = __half22float2(row_x_vec2[i]);
        local_sum += val.x * val.x + val.y * val.y;
    }
    for (int i = tail_start + threadIdx.x; i < hidden; i += BLOCK_SIZE) {
        const float val = __half2float(row_x[i]);
        local_sum += val * val;
    }

    const float sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum / static_cast<float>(hidden) + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < vec_hidden; i += BLOCK_SIZE) {
        const float2 val = __half22float2(row_x_vec2[i]);
        const float2 out = make_float2(
            val.x * inv_rms * __half2float(gamma[i * 2]),
            val.y * inv_rms * __half2float(gamma[i * 2 + 1]));
        row_y_vec2[i] = __float22half2_rn(out);
    }
    for (int i = tail_start + threadIdx.x; i < hidden; i += BLOCK_SIZE) {
        const float val = __half2float(row_x[i]);
        row_y[i] = __float2half_rn(val * inv_rms * __half2float(gamma[i]));
    }
}

}  // namespace

cudaError_t rmsnorm_half(
    const half* input,
    const half* gamma,
    half* output,
    std::size_t rows,
    std::size_t hidden,
    float eps,
    cudaStream_t stream) {
    if (rows == 0 || hidden == 0) {
        return cudaSuccess;
    }

    if (input == nullptr || gamma == nullptr || output == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        hidden > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidValue;
    }

    dim3 block(kBlockSize);
    dim3 grid(static_cast<unsigned int>(rows));
    rmsnorm_half2_kernel<kBlockSize><<<grid, block, 0, stream>>>(
        input,
        gamma,
        output,
        static_cast<int>(rows),
        static_cast<int>(hidden),
        eps);
    return cudaGetLastError();
}

}  // namespace cuda_oplib
