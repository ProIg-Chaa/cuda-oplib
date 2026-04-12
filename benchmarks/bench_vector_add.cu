#include "cuda_oplib/cuda_check.h"
#include "cuda_oplib/vector_add.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    std::size_t numel = 1 << 24;
    int iters = 100;

    if (argc > 1) {
        numel = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    if (argc > 2) {
        iters = std::atoi(argv[2]);
    }

    std::vector<float> host(numel, 1.0f);
    float* d_lhs = nullptr;
    float* d_rhs = nullptr;
    float* d_out = nullptr;

    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_lhs), numel * sizeof(float)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_rhs), numel * sizeof(float)));
    CUDA_OPLIB_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_out), numel * sizeof(float)));
    CUDA_OPLIB_CHECK_CUDA(
        cudaMemcpy(d_lhs, host.data(), numel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OPLIB_CHECK_CUDA(
        cudaMemcpy(d_rhs, host.data(), numel * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        CUDA_OPLIB_CHECK_CUDA(cuda_oplib::vector_add(d_lhs, d_rhs, d_out, numel));
    }
    CUDA_OPLIB_CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_OPLIB_CHECK_CUDA(cudaEventCreate(&start));
    CUDA_OPLIB_CHECK_CUDA(cudaEventCreate(&stop));

    CUDA_OPLIB_CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CUDA_OPLIB_CHECK_CUDA(cuda_oplib::vector_add(d_lhs, d_rhs, d_out, numel));
    }
    CUDA_OPLIB_CHECK_CUDA(cudaEventRecord(stop));
    CUDA_OPLIB_CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_OPLIB_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const double avg_ms = elapsed_ms / static_cast<double>(iters);
    const double bytes = static_cast<double>(numel) * sizeof(float) * 3.0;
    const double gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;

    std::cout << std::fixed << std::setprecision(3)
              << "vector_add numel=" << numel << " iters=" << iters
              << " avg_ms=" << avg_ms << " throughput_GBps=" << gbps << '\n';

    CUDA_OPLIB_CHECK_CUDA(cudaEventDestroy(start));
    CUDA_OPLIB_CHECK_CUDA(cudaEventDestroy(stop));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_lhs));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_rhs));
    CUDA_OPLIB_CHECK_CUDA(cudaFree(d_out));
    return 0;
}
