#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

//两次reduce，第一次求max，第二次求sum，没有warp优化
template <int BLOCK_SIZE>
__global__ void attention_forward_f32_naive_kernel(
    const float* __restrict__ Q,   // [Sq, D]
    const float* __restrict__ K,   // [Sk, D]
    const float* __restrict__ V,   // [Sk, D]
    float* __restrict__ O,         // [Sq, D]
    int Sq,
    int Sk,
    int D){
        int q_row = blockIdx.x;
        if (q_row >= Sq) return;
        int tid = threadIdx.x;
        const float scale = rsqrtf((float)D); 

        extern __shared__ float smem[]; // Shared memory for scores
        __shared__ float buff[BLOCK_SIZE]; // Buffer for reduction
        float* score = smem; // size: Sk


        
        const float* q_ptr = Q + (size_t)q_row * D;
        float* o_ptr = O + (size_t)q_row * D;

        for(int k_row = tid ; k_row < Sk; k_row += BLOCK_SIZE){
            const float* k_ptr = K + (size_t)k_row * D;

            // Compute the dot product between q and k
            float dot_product = 0.0f;
            for(int d = 0; d < D; ++d){
                dot_product += q_ptr[d] * k_ptr[d];
            }

            score[k_row] = dot_product * scale; // Scale the dot product
        }
        __syncthreads();

        // Compute softmax
        float local_max = -INFINITY;
        for(int k_row = tid ; k_row < Sk; k_row += BLOCK_SIZE){
            local_max = fmaxf(local_max, score[k_row]);
        }
        buff[tid] = local_max;
        __syncthreads();

        // Reduce to find the global max
        for(int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2){
            if(tid < offset){
                buff[tid] = fmaxf(buff[tid], buff[tid + offset]);
            }
            __syncthreads();
        }

        float max_score = buff[0];

        // Compute the local sum of exp(scores - max_score)
        float local_sum = 0.0f;
        for(int k_row = tid ; k_row < Sk; k_row += BLOCK_SIZE){
            float exp_score = expf(score[k_row] - max_score);
            local_sum += exp_score;
            score[k_row] = exp_score;
        }
        buff[tid] = local_sum;
        __syncthreads();

        // Reduce to find the global sum
        for(int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2){
            if(tid < offset){
                buff[tid] = buff[tid] + buff[tid + offset];
            }
            __syncthreads();
        }

        float sum_score = buff[0];

        // Compute the final softmax output
        for(int k_row = tid ; k_row < Sk; k_row += BLOCK_SIZE){
            float softmax_score = score[k_row] / sum_score;
            score[k_row] = softmax_score; // Reuse score array to store softmax scores
            
        }
        __syncthreads();

        // Compute the output O = softmax(QK^T) * V
        for(int d = tid; d < D; d += BLOCK_SIZE){
            float output_value = 0.0f;
            for(int k_row = 0; k_row < Sk; ++k_row){    
                const float* v_ptr = V + (size_t)k_row * D;
                output_value += score[k_row] * v_ptr[d];
            }
            o_ptr[d] = output_value;
        }
    }





struct OnlineSfstate {
    float max;
    float exp_sum;
};

__device__ __inline__ OnlineSfstate online_sfstate_init(){
    OnlineSfstate state;
    state.max = -INFINITY;
    state.exp_sum = 0.0f;
    return state;
}

__device__ __inline__ void combine_state(OnlineSfstate& a, const OnlineSfstate& b){
    if(a.max < b.max){
        float exp_diff = expf(a.max - b.max);
        a.max = b.max;
        a.exp_sum = b.exp_sum + a.exp_sum * exp_diff;
    } else {
        float exp_diff = expf(b.max - a.max);
        a.exp_sum = a.exp_sum + b.exp_sum * exp_diff;
    }
}

template <int MAX_FRAG>
struct VecFragment {
    float acc[MAX_FRAG];
    int d_idx[MAX_FRAG];//记录每个fragment对应的维度索引
    int valid;          //表示当前线程实际负责几个元素
};


//注意，这个初始化函数已经约定好了每个线程负责的维度分布，
//要求每个线程跨blocksize遍历维度，这样才能保证combine时不同线程的fragment对应的维度索引完全一致
template <int MAX_FRAG>
__device__ __inline__ void vec_fragment_init(
    VecFragment<MAX_FRAG>& frag,
    int tid,
    int block_size,
    int D)
    {
        frag.valid = 0;
        for(int d = tid; d < D; d += block_size){
            if(frag.valid < MAX_FRAG){
                frag.d_idx[frag.valid] = d;
                frag.acc[frag.valid] = 0.0f;
                frag.valid++;
            }
        }
}

//每个线程维护一个长度为MAX_FRAG的fragment，适用于D较大的情况
template <int MAX_FRAG>
__device__ __inline__ void online_frafs_update(
    OnlineSfstate& state,
    float score,
    VecFragment<MAX_FRAG>& frag,
    const float* __restrict__ V_row  // [Sk, D] 对于每个score对应的V行
){
    if(state.max < score){
        float exp_diff = expf(state.max - score);
        state.exp_sum = state.exp_sum * exp_diff + 1.0f;
        state.max = score;

        for(int i = 0; i < frag.valid; ++i){
            int d = frag.d_idx[i];
            frag.acc[i] = frag.acc[i] * exp_diff + V_row[d];
        }
    } else {
        float exp_diff = expf(score - state.max);
        state.exp_sum += exp_diff;

        for(int i = 0; i < frag.valid; ++i){
            int d = frag.d_idx[i];
            frag.acc[i] += V_row[d] * exp_diff;
        }
    }
}

//写回output时需要根据维度索引写回正确的位置
template <int MAX_FRAG>
__device__ __inline__ void write_back_output(
    float* __restrict__ O_row, // 对于一行query的输出,shape是[1,D]
    const VecFragment<MAX_FRAG>& frag,
    const OnlineSfstate& state
){
    float inv_l = 1.0f / state.exp_sum;

    #pragma unroll
    for(int i = 0; i < frag.valid; ++i){
        int d = frag.d_idx[i];
        O_row[d] = frag.acc[i] * inv_l; // Write back the normalized output
    }  
}


//combine方法需要保证两个状态的fragment对应的维度索引完全一致，
//即每个线程负责的维度在两个状态中都要有，且位置相同，这样才能正确地合并accumulator
template <int MAX_FRAG>
__device__ __inline__ void online_frafs_combine(
    OnlineSfstate& a,
    const OnlineSfstate& b,
    VecFragment<MAX_FRAG>& frag_a,
    const VecFragment<MAX_FRAG>& frag_b
){
    if(a.max < b.max){
        float exp_diff = expf(a.max - b.max);
        a.max = b.max;
        a.exp_sum = b.exp_sum + a.exp_sum * exp_diff;
        for(int i = 0; i < frag_a.valid; ++i){
            int d_a = frag_a.d_idx[i];
            float acc_a = frag_a.acc[i] * exp_diff;
            // Find corresponding fragment in frag_b
            for(int j = 0; j < frag_b.valid; ++j){
                if(frag_b.d_idx[j] == d_a){
                    acc_a += frag_b.acc[j];
                    break;
                }
            }
            frag_a.acc[i] = acc_a;
        }
    } else {
        float exp_diff = expf(b.max - a.max);
        a.exp_sum = a.exp_sum + b.exp_sum * exp_diff;
        for(int i = 0; i < frag_a.valid; ++i){
            int d_a = frag_a.d_idx[i];
            float acc_a = frag_a.acc[i];
            // Find corresponding fragment in frag_b
            for(int j = 0; j < frag_b.valid; ++j){
                if(frag_b.d_idx[j] == d_a){
                    acc_a += frag_b.acc[j] * exp_diff;
                    break;
                }
            }
            frag_a.acc[i] = acc_a;
        }
    }
}


//以下两个方法要求线程维护完整的一行的accumulator，适用于D较小的情况
__device__ __inline__ void online_softmax_update(
    OnlineSfstate& state,
    float score,
    float* acc,
    int D,
    const float* __restrict__ V   // [Sk, D]
){
    if(state.max < score){
        float exp_diff = expf(state.max - score);
        state.exp_sum = state.exp_sum * exp_diff + 1.0f;
        state.max = score;

        for(int d = 0; d < D; ++d){
            acc[d] = acc[d] * exp_diff + V[d];
        }
    } else {
        float exp_diff = expf(score - state.max);
        state.exp_sum += exp_diff;

        for(int d = 0; d < D; ++d){
            acc[d] += V[d] * exp_diff;
        }
    }
}

//同样要求线程维护完整行的accumulator，combine方法更简单直接
__device__ __inline__ void online_softmax_combine(
    OnlineSfstate& a,
    const OnlineSfstate& b,
    float* acc,
    float* acc_other,
    int D
){
    if(a.max < b.max){
        float exp_diff = expf(a.max - b.max);
        a.max = b.max;
        a.exp_sum = b.exp_sum + a.exp_sum * exp_diff;
        for(int d = 0; d < D; ++d){
            acc[d] = acc[d] * exp_diff + acc_other[d];
        }
    } else {
        float exp_diff = expf(b.max - a.max);
        a.exp_sum = a.exp_sum + b.exp_sum * exp_diff;
        for(int d = 0; d < D; ++d){
            acc[d] = acc[d] + acc_other[d] * exp_diff;
        }
    }
}

template <int BLOCK_SIZE,int MAX_FRAG>
__global__ void onlinesfatt_forward_f32_naive_kernel(
    const float* __restrict__ Q,   // [Sq, D]
    const float* __restrict__ K,   // [Sk, D]
    const float* __restrict__ V,   // [Sk, D]
    float* __restrict__ O,         // [Sq, D]
    int Sq,
    int Sk,
    int D){
        int q_row = blockIdx.x;
        if (q_row >= Sq) return;
        int tid = threadIdx.x;

        
        extern __shared__ float smem[]; // Shared memory for scores
        float* score = smem; // size: Sk
        const float scale = rsqrtf((float)D);


        
        const float* q_ptr = Q + (size_t)q_row * D;
        float* o_ptr = O + (size_t)q_row * D;

        for(int k_row = tid ; k_row < Sk; k_row += BLOCK_SIZE){
            const float* k_ptr = K + (size_t)k_row * D;

            // Compute the dot product between q and k
            float dot_product = 0.0f;
            for(int d = 0; d < D; ++d){
                dot_product += q_ptr[d] * k_ptr[d];
            }

            score[k_row] = dot_product * scale; // Scale the dot product
        }
        __syncthreads();

        //每个线程跨blocksize遍历score，在线更新softmax状态
        OnlineSfstate local_state;
        VecFragment<MAX_FRAG> local_frag;
        local_state = online_sfstate_init();
        vec_fragment_init(local_frag, tid, BLOCK_SIZE, D);
        for(int k_row = 0; k_row < Sk; ++k_row){
            const float* v_ptr = V + (size_t)k_row * D;
            online_frafs_update(local_state, score[k_row], local_frag, v_ptr);
        }
        write_back_output(o_ptr, local_frag, local_state);

    }




