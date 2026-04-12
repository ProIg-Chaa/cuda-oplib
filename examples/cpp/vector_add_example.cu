#include "cuda_oplib/cuda_check.h"
#include "cuda_oplib/vector_add.h"

#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t kNumel = 8;

    std::vector<float> lhs(kNumel);
    std::vector<float> rhs(kNumel);
    for (std::size_t i = 0; i < kNumel; ++i) {
        lhs[i] = static_cast<float>(i);
        rhs[i] = static_cast<float>(i * 2);
    }

    float* d_lhs = nullptr;
    float* d_rhs = nullptr;
    float* d_out = nullptr;

    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_lhs), kNumel * sizeof(float)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_rhs), kNumel * sizeof(float)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_out), kNumel * sizeof(float)));

    CUDA_OPLIB_CHECK_CUDA(
        cudaMemcpy(d_lhs, lhs.data(), kNumel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OPLIB_CHECK_CUDA(
        cudaMemcpy(d_rhs, rhs.data(), kNumel * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_OPLIB_CHECK_CUDA(cuda_oplib::vector_add(d_lhs, d_rhs, d_out, kNumel));
    CUDA_OPLIB_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> out(kNumel);
    CUDA_OPLIB_CHECK_CUDA(
        cudaMemcpy(out.data(), d_out, kNumel * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < kNumel; ++i) {
        std::cout << lhs[i] << " + " << rhs[i] << " = " << out[i] << '\n';
    }

    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_lhs));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_rhs));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_out));
    return 0;
}
