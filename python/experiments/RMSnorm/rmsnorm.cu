#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__inline__ __device__ float block_reduce_sum(float val) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
    static __shared__ float shared[BLOCK_SIZE >> 5]; // Assuming max 1024 threads per block
    int lane = threadIdx.x & 31; // Thread index within the warp
    int warpid = threadIdx.x >> 5; // Warp index within the block

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warpid] = val;
    }
    __syncthreads();

    // Only the first warp will perform the final reduction
    if (warpid == 0) {
        val = (threadIdx.x < BLOCK_SIZE >> 5) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}


template <int BLOCK_SIZE>
__global__ void rmsnorm_forward_f32_warp_kernel(
    const float* __restrict__ x,       // [M, N]
    const float* __restrict__ gamma,   // [N]
    float* __restrict__ y,             // [M, N]
    int rows,            
    int N,
    float eps){
        int row = blockIdx.x;
        if (row >= rows) return;

        const float* row_x = x + blockIdx.x * N;
        float* row_y = y + blockIdx.x * N;

        __shared__ float inv_rms;
        float localsum = 0.0f;

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float val = row_x[i];
            localsum += val * val;
        }
        
        float sum = block_reduce_sum<BLOCK_SIZE>(localsum);

        if (threadIdx.x == 0) {
            inv_rms = rsqrtf(sum / N + eps);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float val = row_x[i];
            row_y[i] = val * inv_rms * gamma[i];
        }

    }

template <int BLOCK_SIZE>
__global__ void rmsnorm_forward_h2_warp_kernel(
    const half* __restrict__ x,       // [M, N]
    const half* __restrict__ gamma,   // [N]
    half* __restrict__ y,             // [M, N]
    int rows,
    int hidden,             
    int N,
    float eps){
        int row = blockIdx.x;
        if (row >= rows) return;
        
        const half* row_x = x + blockIdx.x * N;
        half* row_y = y + blockIdx.x * N;

        //检查row_x和row_y是否都满足4字节对齐，如果满足则可以使用half2进行向量化处理
        bool aligned = ((reinterpret_cast<std::uintptr_t>(row_x) & 0x3) == 0) &&
                       ((reinterpret_cast<std::uintptr_t>(row_y) & 0x3) == 0);
        int vecN = aligned ? (N / 2) : 0;//使用vecn作为向量化处理的元素数量，同时巧妙地避免了对齐问题        
        int tail_start = vecN * 2;//如果对齐则vecN是N/2，否则是0，这样tail_start就是N或者0，正好覆盖了所有元素
        
        const half2* row_x_vec2 = aligned ? reinterpret_cast<const half2*>(row_x) : nullptr;
        half2* row_y_vec2 = aligned ? reinterpret_cast<half2*>(row_y) : nullptr;

        __shared__ float inv_rms;
        float localsum = 0.0f;

        for (int i = threadIdx.x; i < vecN; i += blockDim.x) {
            half2 val = row_x_vec2[i];
            float2 fval = __half22float2(val);
            localsum += fval.x * fval.x + fval.y * fval.y;
        }

        for (int i = tail_start + threadIdx.x; i < N; i += blockDim.x) {
            float val = __half2float(row_x[i]);
            localsum += val * val;
        }

        float sum = block_reduce_sum<BLOCK_SIZE>(localsum);
        if (threadIdx.x == 0) {
            inv_rms = rsqrtf(sum / N + eps);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < vecN; i += blockDim.x) {
            half2 val = row_x_vec2[i];
            float2 fval = __half22float2(val);
            float2 res = make_float2(fval.x * inv_rms * __half2float(gamma[i*2]), 
                                      fval.y * inv_rms * __half2float(gamma[i*2 + 1]));
            row_y_vec2[i] = __float22half2_rn(res);
        }

        for (int i = tail_start + threadIdx.x; i < N; i += blockDim.x) {
            float val = __half2float(row_x[i]);
            float res = val * inv_rms * __half2float(gamma[i]);
            row_y[i] = __float2half_rn(res);
        }


    }