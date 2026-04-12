#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_oplib {

cudaError_t vector_add(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t numel,
    cudaStream_t stream = 0);

}  // namespace cuda_oplib

