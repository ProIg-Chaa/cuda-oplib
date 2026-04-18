#include "cuda_oplib/cuda_check.h"
#include "cuda_oplib/rmsnorm.h"

#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t kRows = 2;
    constexpr std::size_t kHidden = 8;

    std::vector<half> x(kRows * kHidden);
    std::vector<half> gamma(kHidden, __float2half(1.0f));

    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = __float2half(static_cast<float>(i + 1));
    }

    half* d_x = nullptr;
    half* d_gamma = nullptr;
    half* d_y = nullptr;

    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_x), x.size() * sizeof(half)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_gamma), gamma.size() * sizeof(half)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_y), x.size() * sizeof(half)));

    CUDA_OPLIB_CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_OPLIB_CHECK_CUDA(
        cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(half), cudaMemcpyHostToDevice));

    CUDA_OPLIB_CHECK_CUDA(cuda_oplib::rmsnorm_half(d_x, d_gamma, d_y, kRows, kHidden, 1e-5f));
    CUDA_OPLIB_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<half> out(x.size());
    CUDA_OPLIB_CHECK_CUDA(cudaMemcpy(out.data(), d_y, out.size() * sizeof(half), cudaMemcpyDeviceToHost));

    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t col = 0; col < kHidden; ++col) {
            std::cout << __half2float(out[row * kHidden + col]) << ' ';
        }
        std::cout << '\n';
    }

    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_x));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_gamma));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_y));
    return 0;
}
