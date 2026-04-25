#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>


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

//在线合并两个softmaxstate，适用于规约时合并warp状态或者block内不同warp状态
__device__ __inline__ void combine_state(OnlineSfstate& a, const OnlineSfstate& b){
    if(a.max == -INFINITY && b.max != -INFINITY) {a.max = b.max; a.exp_sum = b.exp_sum;return;}; // a和b都是空状态，不需要合并
    if(b.max == -INFINITY) return; // b是空状态，不需要合并

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
    int d_idx[MAX_FRAG];//记录每个fragment对应的维度索引,针对v
    int valid;          //表示当前线程实际负责几个元素
};


//注意，这个初始化函数已经约定好了每个线程负责的维度分布，
//要求每个线程跨warpsize遍历维度，这样才能保证combine时不同线程的fragment对应的维度索引完全一致
template <int MAX_FRAG>
__device__ __inline__ void vec_fragment_init(
    VecFragment<MAX_FRAG>& frag,
    int laneid,
    int D)
    {
        frag.valid = 0;
        for(int d = laneid; d < D; d += 32){
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

//这个函数在warp内规约计算点积
__device__ __inline__ float warp_reduce_score(float score){
    for(int offset = 16; offset > 0; offset >>= 1){
        score += __shfl_down_sync(0xffffffff, score, offset);
    }
    return score;
}



//这个函数在warp内规约state
__device__ __inline__ void warp_reduce_state(OnlineSfstate& state){
    for(int offset = 16; offset > 0; offset >>= 1){
        OnlineSfstate other;
        other.max = __shfl_down_sync(0xffffffff, state.max, offset);
        other.exp_sum = __shfl_down_sync(0xffffffff, state.exp_sum, offset);
        combine_state(state, other);
    }
}


//block 内不同 warp 按 k_row 切分，各自处理一部分 key/value tile。
// 对于某个 warp 负责的 tile，warp 内线程共享该 tile 的 score，
//并按输出维度 d 对 value 向量做并行分工，每个线程维护自己负责的输出向量片段。
// 因而每个 warp 最终得到该 tile 的局部 softmax 状态 (m_warp, l_warp, acc_warp)。
template <int BLOCK_SIZE,int MAX_FRAG>
__device__ __inline__ void warp_softmax_online(
    float* __restrict__ q_buff,   // [D]
    const float* __restrict__ K_tile,  // [tile_k, D] 负责的key tile
    const float* __restrict__ V_tile,  // [tile_k, D] 负责的value tile
    VecFragment<MAX_FRAG>& frag,       // 每个线程维护的输出片段
    float &m_warp,
    float &l_warp,
    int tile_k, //tile的k维度大小
    int D,
    int laneid,
    int wid,
    int q_row,
    int debug
){

    __shared__ OnlineSfstate warp_states[BLOCK_SIZE / 32]; // 每个warp一个状态
    __shared__ OnlineSfstate block_state; // block内规约后的状态
    __shared__ float buff[BLOCK_SIZE * MAX_FRAG] ; // 每个线程一个fragment的buffer,用于规约时交换fragment



    float alpha_warp = 0.0f;
    float beta_warp = 0.0f;
    float inv_d = 1.0f / sqrtf((float)D);

    //先计算score并在线更新softmax状态和accumulator
    //把query行加载到共享内存，供warp内线程共享

    __syncthreads(); //确保query加载完成

    for(int k = 0; k < tile_k; ++k){
        //规约计算点积，得到score        
        float score = 0.0f;
        for(int d = laneid; d < D; d += 32){
            score += q_buff[d] * K_tile[k * D + d];
        }

        score = warp_reduce_score(score); //第一个线程得到最终的score
        //把这个score广播给warp内所有线程，供更新状态和accumulator使用
        score = __shfl_sync(0xffffffff, score, 0);
        score *= inv_d;
        //现在只让laneid==0的线程更新状态，其他线程只更新accumulator，保证每个warp内状态的一致性
        if(laneid == 0){
            if (m_warp < score) {
                float scale = expf(m_warp - score);
                l_warp = l_warp * scale + 1.0f;
                m_warp = score;
                alpha_warp = scale;
                beta_warp = 1.0f;
            } else {
                float weight = expf(score - m_warp);
                l_warp += weight;
                alpha_warp = 1.0f;
                beta_warp = weight;
            }            
        }

        m_warp = __shfl_sync(0xffffffff, m_warp, 0);
        l_warp = __shfl_sync(0xffffffff, l_warp, 0);
        alpha_warp = __shfl_sync(0xffffffff, alpha_warp, 0);
        beta_warp  = __shfl_sync(0xffffffff, beta_warp, 0);

        for(int i = 0; i < frag.valid; ++i){
            int d = frag.d_idx[i];
            float v_val = V_tile[k * D + d];
            frag.acc[i] = frag.acc[i] * alpha_warp + v_val * beta_warp;
        }

        if (debug && q_row == 0 && laneid == 0) {
            printf("[warp-local] wid=%d k=%d score=%f max=%f exp_sum=%f\n",
                   wid, k, score, m_warp, l_warp);
        }
    }
    

}



template <int BLOCK_SIZE,int MAX_FRAG>
__global__ void onlinesfatt_forward_f32_warp_kernel(
    const float* __restrict__ Q,   // [Sq, D]
    const float* __restrict__ K,   // [Sk, D]
    const float* __restrict__ V,   // [Sk, D]
    float* __restrict__ O,         // [Sq, D]
    int Sq,
    int Sk,
    int D,
    int debug){

    int q_row = blockIdx.x; // 每个block处理一个query行
    if(q_row >= Sq) return;
    extern __shared__ float q_buff[] ; // 每行query的缓存，供warp内线程共享，避免重复访问全局内存
    for(int d = threadIdx.x; d < D; d += BLOCK_SIZE){
        q_buff[d] = Q[q_row * D + d];
    }
    __syncthreads(); //确保query加载完成

    int tid = threadIdx.x; // 每个线程处理一个维度片段
    int wid = tid / 32; // Assuming warp size is 32
    int laneid = tid % 32;
    int warp_num = BLOCK_SIZE / 32;

    float m_warp = -INFINITY; //每个warp维护一个局部的softmax状态
    float l_warp = 0.0f;

    float* O_row = O + (size_t)q_row * D; // 输出行指针
    
    
    VecFragment<MAX_FRAG> frag;
    vec_fragment_init(frag, laneid, D);
    OnlineSfstate state = online_sfstate_init();

    
    //按k维切分，每个warp处理一个tile
    //所有warp都参与计算，但不同warp负责不同的tile
    int tile_k_size = 16; //每个tile的k维度大小，可以调整这个参数来权衡计算和规约的开销
    for(int k_start = wid * tile_k_size; k_start < Sk; k_start += warp_num * tile_k_size){
        int k_end = min(k_start + tile_k_size, Sk);
        warp_softmax_online<BLOCK_SIZE, MAX_FRAG>(q_buff, K + k_start * D, V + k_start * D, frag,
                                                m_warp, l_warp,  k_end - k_start, D, laneid, wid, q_row, debug);
    }

    __shared__ OnlineSfstate warp_states[BLOCK_SIZE / 32]; // 每个warp一个状态
    __shared__ OnlineSfstate block_state; // block内规约后的状态
    __shared__ float buff[BLOCK_SIZE * MAX_FRAG] ; // 每个线程一个fragment的buffer,用于规约时交换fragment

    if(laneid == 0){ // 每个warp的第一个线程写回状态到共享内存
        warp_states[threadIdx.x / 32].max = m_warp;
        warp_states[threadIdx.x / 32].exp_sum = l_warp;
    }
    
    //开始对所有block内的所有warp状态规约
    __syncthreads(); 
    int warpnum = BLOCK_SIZE / 32;
    //前warpnum个线程加载一个warp的状态，其他线程不需要加载
    if(threadIdx.x < warpnum){
        state = warp_states[threadIdx.x]; //每个线程加载一个warp的状态
    } else {
        state = online_sfstate_init(); //其他线程初始化一个空状态   
    }
    __syncthreads(); 
    
    //规约状态
    warp_reduce_state(state);//第一个线程得到全局规约后的状态
    if(threadIdx.x == 0){
        block_state = state; //写回规约后的状态到共享内存
    }
    __syncthreads(); 

    state = warp_states[wid];
    if (debug && q_row == 0 && laneid == 0) {
        printf("[warp-state] wid=%d local_max=%f local_exp_sum=%f block_max=%f block_exp_sum=%f\n",
               wid, state.max, state.exp_sum, block_state.max, block_state.exp_sum);
    }

    //每个线程加载规约后的状态
    float diff = expf(state.max - block_state.max); //规约后的max和当前线程状态的max的差值
    
    for(int i = 0; i < frag.valid; ++i){        
        frag.acc[i] = frag.acc[i] * diff / block_state.exp_sum; //把每个线程的accumulator都调整到规约后的max上
        if (debug && q_row == 0 && wid == 0 && laneid < 2) {
            printf("[frag-scaled] wid=%d lane=%d i=%d d=%d acc=%f\n",
                   wid, laneid, i, frag.d_idx[i], frag.acc[i]);
        }
    }

    //每个线程把自己的fragment写到共享内存中，准备规约fragment
    int offset = threadIdx.x * MAX_FRAG; //按warp和线程索引计算在buff中的偏移，每个线程一个连续的片段
    for(int i = 0; i < frag.valid; ++i){
        buff[offset + i] = frag.acc[i];//有可能存在valid < MAX_FRAG的情况
    }
    for(int i = frag.valid; i < MAX_FRAG; ++i){
        buff[offset + i] = 0.0f; //如果valid < MAX_FRAG,把多余的部分填0，规约时就不会对结果产生影响
    }
    __syncthreads();

    //warp0负责规约所有线程的fragment到一个最终的输出
    //warp 0 的每个 lane 读所有 warp 相同 lane 的 fragment
    if(wid == 0){
        for(int i = 0; i < frag.valid; ++i){ //每个线程负责一个输出维度的片段
            int d = frag.d_idx[i];
            float acc = 0.0f;
            for(int w = 0; w < warpnum; ++w){
                acc += buff[w * 32 * MAX_FRAG + laneid * MAX_FRAG + i]; //累加所有warp相同lane的fragment
            }
            O_row[d] = acc; //写回输出
            if (debug && q_row == 0 && laneid < 2) {
                printf("[output-write] lane=%d i=%d d=%d out=%f\n", laneid, i, d, acc);
            }
        }
    }

}
    
    
