#include "cuda_oplib/cuda_check.h"
#include "cuda_oplib/rmsnorm.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

void reference_rmsnorm(
    const std::vector<half>& x,
    const std::vector<half>& gamma,
    std::vector<float>& out,
    std::size_t rows,
    std::size_t hidden,
    float eps) {
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * hidden;
        float sumsq = 0.0f;
        for (std::size_t i = 0; i < hidden; ++i) {
            const float val = __half2float(x[base + i]);
            sumsq += val * val;
        }

        const float inv_rms =
            1.0f / std::sqrt(sumsq / static_cast<float>(hidden) + eps);
        for (std::size_t i = 0; i < hidden; ++i) {
            out[base + i] =
                __half2float(x[base + i]) * inv_rms * __half2float(gamma[i]);
        }
    }
}

}  // namespace

int main() {
    constexpr std::size_t kRows = 3;
    constexpr std::size_t kHidden = 7;
    constexpr float kEps = 1e-5f;

    std::vector<half> x(kRows * kHidden);
    std::vector<half> gamma(kHidden);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t col = 0; col < kHidden; ++col) {
            x[row * kHidden + col] =
                __float2half(static_cast<float>(0.2f * (row + 1) + 0.35f * col));
        }
    }
    for (std::size_t i = 0; i < kHidden; ++i) {
        gamma[i] = __float2half(0.8f + static_cast<float>(i) * 0.05f);
    }

    half* d_x = nullptr;
    half* d_gamma = nullptr;
    half* d_y = nullptr;

    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_x), x.size() * sizeof(half)));
    CUDA_OPLIB_CHECK_CUDA(
        cudaMalloc(reinterpret_cast<void**>(&d_gamma), gamma.size() * sizeof(half)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_y), x.size() * sizeof(half)));

    CUDA_OPLIB_CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_OPLIB_CHECK_CUDA(
        cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(half), cudaMemcpyHostToDevice));

    CUDA_OPLIB_CHECK_CUDA(cuda_oplib::rmsnorm_half(d_x, d_gamma, d_y, kRows, kHidden, kEps));
    CUDA_OPLIB_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<half> out_half(x.size());
    CUDA_OPLIB_CHECK_CUDA(
        cudaMemcpy(out_half.data(), d_y, out_half.size() * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<float> expected(x.size(), 0.0f);
    reference_rmsnorm(x, gamma, expected, kRows, kHidden, kEps);

    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_x));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_gamma));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_y));

    for (std::size_t i = 0; i < out_half.size(); ++i) {
        const float got = __half2float(out_half[i]);
        const float diff = std::fabs(got - expected[i]);
        if (diff > 2e-2f) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << expected[i]
                      << ", got " << got
                      << ", diff " << diff << '\n';
            return 1;
        }
    }

    std::cout << "rmsnorm_half smoke test passed\n";
    return 0;
}
