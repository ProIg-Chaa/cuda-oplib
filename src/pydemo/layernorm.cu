#include <cuda_runtime.h>
#include <stdio.h>
#include<math.h>


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