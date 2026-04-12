#include "cuda_oplib/vector_add.h"

namespace cuda_oplib {
namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void VectorAddKernel(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t numel) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = lhs[idx] + rhs[idx];
    }
}

}  // namespace

cudaError_t vector_add(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t numel,
    cudaStream_t stream) {
    if (numel == 0) {
        return cudaSuccess;
    }

    if (lhs == nullptr || rhs == nullptr || out == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    const int blocks =
        static_cast<int>((numel + kThreadsPerBlock - 1) / kThreadsPerBlock);
    VectorAddKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(lhs, rhs, out, numel);
    return cudaGetLastError();
}

}  // namespace cuda_oplib

