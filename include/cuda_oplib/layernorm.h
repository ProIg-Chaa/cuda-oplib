#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_oplib {

cudaError_t layernorm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t rows,
    std::size_t hidden,
    float eps,
    cudaStream_t stream = 0);

}  // namespace cuda_oplib
