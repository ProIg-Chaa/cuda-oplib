#include "cuda_oplib/cuda_check.h"
#include "cuda_oplib/vector_add.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t kNumel = 1024;
    std::vector<float> lhs(kNumel);
    std::vector<float> rhs(kNumel);
    std::vector<float> expected(kNumel);

    for (std::size_t i = 0; i < kNumel; ++i) {
        lhs[i] = static_cast<float>(i % 37);
        rhs[i] = static_cast<float>((i * 3) % 17);
        expected[i] = lhs[i] + rhs[i];
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

    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_lhs));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_rhs));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_out));

    for (std::size_t i = 0; i < kNumel; ++i) {
        if (std::fabs(out[i] - expected[i]) > 1e-6f) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i]
                      << ", got " << out[i] << '\n';
            return 1;
        }
    }

    std::cout << "vector_add smoke test passed\n";
    return 0;
}
