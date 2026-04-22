#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

__inline__ __device__ float fused_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__inline__ __device__ float fused_block_reduce_sum(float val) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
    static __shared__ float shared[BLOCK_SIZE >> 5]; // Assuming max 1024 threads per block
    int lane = threadIdx.x & 31; // Thread index within the warp
    int warpid = threadIdx.x >> 5; // Warp index within the block

    val = fused_warp_reduce_sum(val);

    if (lane == 0) {
        shared[warpid] = val;
    }
    __syncthreads();

    // Only the first warp will perform the final reduction
    if (warpid == 0) {
        val = (threadIdx.x < BLOCK_SIZE >> 5) ? shared[lane] : 0.0f;
        val = fused_warp_reduce_sum(val);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_forward_f32_warp_kernel(
    const float* __restrict__ x,            // [B, N]
    const float* __restrict__ gamma,        // [N]
    const float* __restrict__ residual,     // [B, N]
    float* __restrict__ y,                  // [B, N]
    int rows,
    int N,
    float eps) {
        int row = blockIdx.x;
        if (row >= rows) return;
        
        const float* row_x = x + row * N;
        const float* row_residual = residual + row * N;
        float* row_y = y + row * N;

        __shared__ float invsqrt_var;

        float localsum = 0.0f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float val = row_x[i] + row_residual[i];
            localsum += val * val;
        }

        float sum = fused_block_reduce_sum<BLOCK_SIZE>(localsum);

        if (threadIdx.x == 0) {
            invsqrt_var = rsqrtf(sum / N + eps);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float val = row_x[i] + row_residual[i];
            row_y[i] = val * invsqrt_var * gamma[i];
        }
}

template <int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_forward_h2_warp(
    const half* __restrict__ x,            // [B, N]
    const half* __restrict__ gamma,        // [N]
    const half* __restrict__ residual,     // [B, N]
    half* __restrict__ y,                  // [B, N]
    int rows,
    int N,
    float eps) {
        int row = blockIdx.x;
        if (row >= rows) return;

        const half* row_x = x + row * N;
        const half* row_residual = residual + row * N;
        half* row_y = y + row * N;

        //检查输入字节对齐
        bool aligned = ((reinterpret_cast<std::uintptr_t>(row_x) & 0x3) == 0) &&
                       ((reinterpret_cast<std::uintptr_t>(row_y) & 0x3) == 0)&&
                       ((reinterpret_cast<std::uintptr_t>(row_residual) & 0x3) == 0);
        int vecN = aligned ? (N / 2) : 0;//使用vecn作为向量化处理的元素数量，同时巧妙地避免了对齐问题        
        int tail_start = vecN * 2;//如果对齐则vecN是N/2，否则是0，这样tail_start就是N或者0，正好覆盖了所有元素

        const half2* row_x_vec2 = aligned ? reinterpret_cast<const half2*>(row_x) : nullptr;
        const half2* row_residual_vec2 = aligned ? reinterpret_cast<const half2*>(row_residual) : nullptr;
        half2* row_y_vec2 = aligned ? reinterpret_cast<half2*>(row_y) : nullptr;

        __shared__ float invsqrt_var;
        float localsum = 0.0f;

        //向量化处理对齐的部分
        for(int i = threadIdx.x; i < vecN; i += blockDim.x) {
            half2 x_val = row_x_vec2[i];
            half2 residual_val = row_residual_vec2[i];

            float2 a = __half22float2(x_val);
            float2 b = __half22float2(residual_val);
            float2 val_f2 = make_float2(a.x + b.x, a.y + b.y);
            localsum += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
        }

        //处理剩余的部分
        for(int i = tail_start + threadIdx.x; i < N; i += blockDim.x) {
            float val = __half2float(row_x[i]) + __half2float(row_residual[i]);
            localsum += val * val;
        }

        float sum = fused_block_reduce_sum<BLOCK_SIZE>(localsum);
        if (threadIdx.x == 0) {
            invsqrt_var = rsqrtf(sum / N + eps);
        }
        __syncthreads();

        //向量化处理对齐的部分
        for(int i = threadIdx.x; i < vecN; i += blockDim.x) {
            half2 x_val = row_x_vec2[i];
            half2 residual_val = row_residual_vec2[i];
            float2 a = __half22float2(x_val);
            float2 b = __half22float2(residual_val);

            float2 val_f2 = make_float2(a.x + b.x, a.y + b.y);
            float2 y_val_f2 = make_float2(val_f2.x * invsqrt_var * __half2float(gamma[i*2]), val_f2.y * invsqrt_var * __half2float(gamma[i*2+1]));
            row_y_vec2[i] = __float22half2_rn(y_val_f2);
        }

        for(int i = tail_start + threadIdx.x; i < N; i += blockDim.x) {
            float val = __half2float(row_x[i]) + __half2float(row_residual[i]);
            row_y[i] = __float2half(val * invsqrt_var * __half2float(gamma[i]));
        }

    }
