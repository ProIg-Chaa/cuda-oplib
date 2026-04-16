#include "cuda_oplib/cuda_check.h"
#include "cuda_oplib/layernorm.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    std::size_t rows = 4096;
    std::size_t hidden = 768;
    int iters = 200;

    if (argc > 1) {
        rows = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    if (argc > 2) {
        hidden = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    }
    if (argc > 3) {
        iters = std::atoi(argv[3]);
    }

    const std::size_t numel = rows * hidden;
    std::vector<half> host_x(numel, __float2half(1.0f));
    std::vector<half> host_gamma(hidden, __float2half(1.0f));
    std::vector<half> host_beta(hidden, __float2half(0.0f));

    half* d_x = nullptr;
    half* d_gamma = nullptr;
    half* d_beta = nullptr;
    half* d_y = nullptr;

    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_x), numel * sizeof(half)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_gamma), hidden * sizeof(half)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_beta), hidden * sizeof(half)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_y), numel * sizeof(half)));

    CUDA_OPLIB_CHECK_CUDA(cudaMemcpy(d_x, host_x.data(), numel * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_OPLIB_CHECK_CUDA(cudaMemcpy(d_gamma, host_gamma.data(), hidden * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_OPLIB_CHECK_CUDA(cudaMemcpy(d_beta, host_beta.data(), hidden * sizeof(half), cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        CUDA_OPLIB_CHECK_CUDA(
            cuda_oplib::layernorm_half(d_x, d_gamma, d_beta, d_y, rows, hidden, 1e-5f));
    }
    CUDA_OPLIB_CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_OPLIB_CHECK_CUDA(cudaEventCreate(&start));
    CUDA_OPLIB_CHECK_CUDA(cudaEventCreate(&stop));

    CUDA_OPLIB_CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CUDA_OPLIB_CHECK_CUDA(
            cuda_oplib::layernorm_half(d_x, d_gamma, d_beta, d_y, rows, hidden, 1e-5f));
    }
    CUDA_OPLIB_CHECK_CUDA(cudaEventRecord(stop));
    CUDA_OPLIB_CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_OPLIB_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const double avg_ms = elapsed_ms / static_cast<double>(iters);
    const double bytes =
        static_cast<double>(numel) * sizeof(half) * 2.0 +
        static_cast<double>(hidden) * sizeof(half) * 2.0 +
        static_cast<double>(numel) * sizeof(half);
    const double gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;

    std::cout << std::fixed << std::setprecision(3)
              << "layernorm_half rows=" << rows
              << " hidden=" << hidden
              << " iters=" << iters
              << " avg_ms=" << avg_ms
              << " approx_throughput_GBps=" << gbps << '\n';

    CUDA_OPLIB_CHECK_CUDA(cudaEventDestroy(start));
    CUDA_OPLIB_CHECK_CUDA(cudaEventDestroy(stop));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_x));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_gamma));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_beta));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_y));
    return 0;
}
