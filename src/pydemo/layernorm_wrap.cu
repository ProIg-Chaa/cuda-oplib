#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Reuse the existing kernels without modifying layernorm.cu.
#define main layernorm_demo_main
#include "layernorm.cu"
#undef main

namespace {

torch::Tensor launch_wrap(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    auto x_ = x.contiguous();
    auto gamma_ = gamma.contiguous();
    auto beta_ = beta.contiguous();
    auto y = torch::empty_like(x_);

    int rows = static_cast<int>(x_.size(0));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    layernorm_wrap_kernel<<<rows, threads, 0, stream>>>(
        x_.data_ptr<float>(),
        gamma_.data_ptr<float>(),
        beta_.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        static_cast<int>(x_.size(1)),
        static_cast<float>(eps));

    return y;
}

}  // namespace

torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    return launch_wrap(x, gamma, beta, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_wrap", &forward_wrap, "Wrap warp-reduction layernorm CUDA kernel");
}
