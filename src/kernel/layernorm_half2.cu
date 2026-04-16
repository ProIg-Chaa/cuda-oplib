#include "cuda_oplib/layernorm.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cuda_oplib {
    namespace
    {
    struct WelfordData{
        float mean;
        float M2;
        int count;
    };

    __device__ __forceinline__ WelfordData welford_update(WelfordData& state, float x){
        state.count += 1;
        float delta = x - state.mean;
        state.mean += delta / state.count;
        float delta2 = x - state.mean;
        state.M2 += delta * delta2;
        return state;
    }

    //向量化版本的welford_update，处理float4类型的数据,仅向量化读，无向量化数学运算
    __device__ __forceinline__ WelfordData welford_update_vec4(WelfordData& state, float4 x){
        state = welford_update(state, x.x);
        state = welford_update(state, x.y);
        state = welford_update(state, x.z);
        state = welford_update(state, x.w);
        return state;
    }
    //处理float2类型的数据
    __device__ __forceinline__ WelfordData welford_update_vec2(WelfordData& state, float2 x){
        state = welford_update(state, x.x);
        state = welford_update(state, x.y);
        return state;
    }



    __device__ __forceinline__ WelfordData welford_combine(const WelfordData& a, const WelfordData& b){
        WelfordData combined;

        if (a.count == 0) return b;
        if (b.count == 0) return a;
        
        combined.count = a.count + b.count;
        float delta = b.mean - a.mean;

        combined.mean = (a.mean * a.count + b.mean * b.count) / combined.count;
        combined.M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / combined.count;
        return combined;
    }

    __device__ __forceinline__ void welford_warp_reduce(WelfordData& local_val){
        for(int offset = warpSize / 2; offset > 0; offset /= 2){
            int lane = threadIdx.x & 31;
            WelfordData other;

            other.mean = __shfl_down_sync(0xffffffff, local_val.mean, offset);
            other.M2 = __shfl_down_sync(0xffffffff, local_val.M2, offset);
            other.count = __shfl_down_sync(0xffffffff, local_val.count, offset);

            if (lane < offset) {//
                local_val = welford_combine(local_val, other);
            }
        }

    }

    __device__ __forceinline__ void welford_block_reduce(WelfordData& local_val, WelfordData* shared_data){
        int lane = threadIdx.x % warpSize;
        int wrapid = threadIdx.x / warpSize;

        welford_warp_reduce(local_val);

        if (lane == 0) {//只有每个warp的第0个线程有有效的local_val值
            shared_data[wrapid] = local_val;//写入共享内存
        }
        __syncthreads();
        
        if (wrapid == 0) {
            local_val = (lane < blockDim.x / warpSize) ? shared_data[lane] : WelfordData{0.0f, 0.0f, 0};//前warpnum个线程有有效的local_val值
            welford_warp_reduce(local_val);//最后得到全局的WelfordData结果,第一个warp的第0个线程拿到正确的结果
        }
        //注意此处不需要__syncthreads();因为没有跨warp通信了
    }

    __global__ void layernorm_welford_kernel(
        const float* x,
        const float* gamma,
        const float* beta,
        float* y,
        int M,
        int N,
        float eps
    ){
        int row = blockIdx.x;
        int tid = threadIdx.x;

        if (row >= M) return;

        const float* row_x = x + row * N;
        float* row_y = y + row * N;

        __shared__ WelfordData shared_data[32]; // assuming blockDim.x <= 1024, store one WelfordData per warp
        __shared__ float mean;
        __shared__ float var;
        WelfordData local_state = {0.0f, 0.0f, 0};

        for(int i = tid; i < N; i += blockDim.x){
            local_state = welford_update(local_state, row_x[i]);//每个线程跨块处理多个元素，更新自己的WelfordData状态
        }

        welford_block_reduce(local_state, shared_data);//每块第0个线程得到全局的WelfordData结果

        if(tid == 0){
            mean = local_state.mean;
            var = local_state.M2 / local_state.count;
        }
        __syncthreads();
        float inv_std = rsqrtf(var + eps);
        for(int i = tid; i < N; i += blockDim.x){
            float x_hat = (row_x[i] - mean) * inv_std;
            row_y[i] = x_hat * gamma[i] + beta[i];
        }    
    }


    __global__ void layernorm_half2_kernel(
        const half* x,
        const half* gamma,
        const half* beta,
        half* y,
        int M,
        int N,
        float eps
    ){
            int row = blockIdx.x;
            int tid = threadIdx.x;
        
            if (row >= M) return;
        
            const half* row_x = x + row * N;
            half* row_y = y + row * N;

            //检查row_x和row_y是否都满足4字节对齐，如果满足则可以使用half2进行向量化处理
            bool aligned = ((reinterpret_cast<std::uintptr_t>(row_x) & 0x3) == 0) &&
                        ((reinterpret_cast<std::uintptr_t>(row_y) & 0x3) == 0);
            int vecN = aligned ? (N / 2) : 0;//使用vecn作为向量化处理的元素数量，同时巧妙地避免了对齐问题
            
            int tail_start = vecN * 2;//如果对齐则vecN是N/2，否则是0，这样tail_start就是N或者0，正好覆盖了所有元素
            const half2* row_x_vec2 = aligned ? reinterpret_cast<const half2*>(row_x) : nullptr;
            half2* row_y_vec2 = aligned ? reinterpret_cast<half2*>(row_y) : nullptr;

            __shared__ float mean;
            __shared__ float var;
            __shared__ WelfordData shared_data[32];

            WelfordData local_state = {0.0f, 0.0f, 0};
            for(int i = tid; i < vecN; i += blockDim.x){
                local_state = welford_update_vec2(local_state, __half22float2(row_x_vec2[i]));//每个线程跨块处理多个元素，更新自己的WelfordData状态
            }
            for(int i = tail_start + tid; i < N; i += blockDim.x){
                local_state = welford_update(local_state, __half2float(row_x[i]));
            }

            welford_block_reduce(local_state, shared_data);//每块第0个线程得到全局的WelfordData结果

            if(tid == 0){
                mean = local_state.mean;
                var = local_state.M2 / local_state.count;
            }
            __syncthreads();

            float inv_std = rsqrtf(var + eps);
            for(int i = tid; i < vecN; i += blockDim.x){
                float2 x_vec2 = __half22float2(row_x_vec2[i]);
                float2 x_hat_vec2;
                x_hat_vec2.x = (x_vec2.x - mean) * inv_std;
                x_hat_vec2.y = (x_vec2.y - mean) * inv_std;

                float gamma_x = __half2float(gamma[i * 2]);
                float gamma_y = __half2float(gamma[i * 2 + 1]);
                float beta_x = __half2float(beta[i * 2]);
                float beta_y = __half2float(beta[i * 2 + 1]);

                float2 y_vec2;
                y_vec2.x = x_hat_vec2.x * gamma_x + beta_x;
                y_vec2.y = x_hat_vec2.y * gamma_y + beta_y;

                row_y_vec2[i] = __float22half2_rn(y_vec2);

            }
            for(int i = tail_start + tid; i < N; i += blockDim.x){
                float x_hat = (__half2float(row_x[i]) - mean) * inv_std;
                row_y[i] = __float2half(x_hat * __half2float(gamma[i]) + __half2float(beta[i]));
            }
    }
    }//namespace

    cudaError_t layernorm_half(
        const half* x,
        const half* gamma,
        const half* beta,
        half* y,
        std::size_t rows,
        std::size_t hidden,
        float eps,
        cudaStream_t stream
    ){
        if (rows == 0 || hidden == 0) {
            return cudaSuccess;
        }

        if (x == nullptr || gamma == nullptr || beta == nullptr || y == nullptr) {
            return cudaErrorInvalidDevicePointer;
        }

        if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            hidden > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return cudaErrorInvalidValue;
        }

        dim3 block(256);
        dim3 grid(static_cast<unsigned int>(rows));
        layernorm_half2_kernel<<<grid, block, 0, stream>>>(
            x,
            gamma,
            beta,
            y,
            static_cast<int>(rows),
            static_cast<int>(hidden),
            eps);
        return cudaGetLastError();
    }






}//namespace cuda_oplib
