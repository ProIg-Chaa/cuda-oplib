#include <cuda_runtime.h>
#include <stdio.h>
#include<math.h>
#include <cuda_fp16.h>
#include <cstdint>


__global__ void layernorm_naive_kernel(
    const float*  x, 
    const float*  gamma, 
    const float*  beta,
    float*  y, 
    int M, 
    int N,
    float eps
){
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row >= M || col >= N) return;

    const float* x_row = x + row * N;
    float* y_row = y + row * N;

    __shared__ float mean;
    __shared__ float var;

    if (col == 0){
        // compute mean
        float sum = 0.0f;
        for (int i = 0; i < N; ++i){
            sum += x_row[i];
        }
        mean = sum / N;

        // compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < N; ++i){
            var_sum += (x_row[i] - mean) * (x_row[i] - mean);
        }
        var = var_sum / N;
    }
    __syncthreads();

    float x_hat = (x_row[col] - mean) / sqrtf(var + eps);
    y_row[col] = x_hat * gamma[col] + beta[col];
}

__global__ void layernorm_reduction_kernel(
    const float*  x, 
    const float*  gamma, 
    const float*  beta,
    float*  y, 
    int M, 
    int N,
    float eps
){
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= M || col >= N) return;

    const float* row_x = x + row * N;
    float* row_y = y + row * N;

    __shared__ float mean;
    __shared__ float var;
    __shared__ float buf[1024];

    buf[col] = row_x[col];
    __syncthreads();

    for(int stride =N/2; stride > 0; stride >>= 1){
        if(col < stride){
            buf[col] += buf[col + stride];
        }
        __syncthreads();
    }
    if(col == 0){
        mean = buf[0] / N;
    }
    __syncthreads();

    float diff = row_x[col] - mean;
    buf[col] = diff * diff;
    __syncthreads();

    for(int stride = N/2; stride > 0; stride >>= 1){
        if(col < stride){
            buf[col] += buf[col + stride];
        }
        __syncthreads();
    }
    if(col == 0){
        var = buf[0] / N;
    }
    __syncthreads();

    float x_hat = (row_x[col] - mean) / sqrtf(var + eps);
    row_y[col] = x_hat * gamma[col] + beta[col];


}





//1. reduction 还是 shared memory 版

// 每轮都有：

// __syncthreads()

// 这会有同步开销。

// 2. mean 和 var 仍然是两次遍历

// 也就是：

// 第一遍算 mean
// 第二遍算 var
// 第三遍写输出

// global memory 访问仍然很多。

// 3. shared memory reduction 在后半段效率一般

// 特别是 stride 变到 32 以下时，其实可以用 warp-level primitive 代替。

// 4. 访存还没有向量化

// 现在还是一个 float 一个 float 地读。
__global__ void layernorm_stride_kernel(
    const float* x,      // [M, N]
    const float* gamma,  // [N]
    const float* beta,   // [N]
    float* y,            // [M, N]
    int M,
    int N,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) return;

    const float* row_x = x + row * N;
    float* row_y = y + row * N;

    __shared__ float mean;
    __shared__ float var;
    __shared__ float buf[256];


    float locsum = 0.0f;
    for(int i = tid; i < N; i += blockDim.x){
        locsum += row_x[i];
    }
    buf[tid] = locsum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        mean = buf[0] / N; 
    }
    __syncthreads();

    // compute variance and store in buf 
    float locvar = 0.0f;
    for(int i = tid; i < N; i += blockDim.x){
        float diff = row_x[i] - mean;
        locvar += diff * diff;
    }
    buf[tid] = locvar;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        var = buf[0] / N;
    }
    __syncthreads();

    float inv_std = rsqrtf(var + eps);
    for(int i = tid; i < N; i += blockDim.x){
        float x_hat = (row_x[i] - mean) * inv_std;
        row_y[i] = x_hat * gamma[i] + beta[i];
    }


}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32]; // assuming blockDim.x <= 1024,store one reduction result per warp
    int lane = threadIdx.x % warpSize;
    int wrapid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wrapid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    //现在前wrapnum个线程有有效的val
    if (wrapid == 0) {//从正确性来看这一步不必要，但是能节省性能开销
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void layernorm_wrap_kernel(
    const float* x,      // [M, N]
    const float* gamma,  // [N]
    const float* beta,   // [N]
    float* y,            // [M, N]
    int M,
    int N,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) return;

    const float* row_x = x + row * N;//注意这里是乘以N不是blockDim.x，因为每块处理一行的值
    float* row_y = y + row * N;

    __shared__ float mean;
    __shared__ float var;

    float locsum = 0.0f;//行内跨块求和，每个线程处理多个元素，最后通过block_reduce_sum求和，得到每块的局部和，再由每块第0个线程求全局和
    for(int i = tid; i < N; i += blockDim.x){//每个线程跨块处理多个元素
        locsum += row_x[i];
    }

    float sum = block_reduce_sum(locsum);//实际上只有每块第0个线程能拿到正确的sum值
    
    if(tid == 0){//每块处理一行的值，现在每块第0个线程拿到正确的sum值，并写入共享内存
        mean = sum / N; 
    }
    __syncthreads();

    float locvar = 0.0f;//储存的是每块的局部方差
    for(int i = tid; i < N; i += blockDim.x){//依旧是跨块处理多个元素
        float diff = row_x[i] - mean;
        locvar += diff * diff;
    }
    float var_sum = block_reduce_sum(locvar);//每块第0个线程拿到正确的var_sum值
    if(tid == 0){
        var = var_sum / N;
    }
    __syncthreads();

    float inv_std = rsqrtf(var + eps);
    for(int i = tid; i < N; i += blockDim.x){
        float x_hat = (row_x[i] - mean) * inv_std;
        row_y[i] = x_hat * gamma[i] + beta[i];
    }
}

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





int main(){
    const int M = 2;
    const int N = 4;
    float h_x[M * N] = {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f};
    float h_gamma[N] = {1.0f, 1.0f, 1.0f, 1.0f};
    float h_beta[N] = {0.0f, 0.0f, 0.0f, 0.0f};
    float h_y[M * N];

    float *d_x, *d_gamma, *d_beta, *d_y;
    cudaMalloc(&d_x, M * N * sizeof(float));
    cudaMalloc(&d_gamma, N * sizeof(float));
    cudaMalloc(&d_beta, N * sizeof(float));
    cudaMalloc(&d_y, M * N * sizeof(float));

    cudaMemcpy(d_x, h_x, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, N * sizeof(float), cudaMemcpyHostToDevice);

    layernorm_naive_kernel<<<M, N>>>(d_x, d_gamma, d_beta, d_y, M, N, 1e-5);

    cudaMemcpy(h_y, d_y, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            printf("%.4f ", h_y[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_y);

    return 0;
}
