#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace cuda_oplib {

cudaError_t layernorm_half(
    const half* input,
    const half* gamma,
    const half* beta,
    half* output,
    std::size_t rows,
    std::size_t hidden,
    float eps,
    cudaStream_t stream = 0);

}  // namespace cuda_oplib
