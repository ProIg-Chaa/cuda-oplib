#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Reuse the existing kernels without modifying layernorm.cu.
#define main layernorm_demo_main
#include "layernorm.cu"
#undef main

namespace {

void check_float_inputs(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "beta must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D");
    TORCH_CHECK(x.size(1) == gamma.size(0), "hidden dim must match gamma");
    TORCH_CHECK(x.size(1) == beta.size(0), "hidden dim must match beta");
}

void check_half_inputs(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat16, "gamma must be float16");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat16, "beta must be float16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D");
    TORCH_CHECK(x.size(1) == gamma.size(0), "hidden dim must match gamma");
    TORCH_CHECK(x.size(1) == beta.size(0), "hidden dim must match beta");
}

torch::Tensor launch_welford(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    check_float_inputs(x, gamma, beta);
    auto x_ = x.contiguous();
    auto gamma_ = gamma.contiguous();
    auto beta_ = beta.contiguous();
    auto y = torch::empty_like(x_);

    int rows = static_cast<int>(x_.size(0));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    layernorm_welford_kernel<<<rows, threads, 0, stream>>>(
        x_.data_ptr<float>(),
        gamma_.data_ptr<float>(),
        beta_.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        static_cast<int>(x_.size(1)),
        static_cast<float>(eps));

    return y;
}

torch::Tensor launch_half2(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    check_half_inputs(x, gamma, beta);
    auto x_ = x.contiguous();
    auto gamma_ = gamma.contiguous();
    auto beta_ = beta.contiguous();
    auto y = torch::empty_like(x_);

    int rows = static_cast<int>(x_.size(0));
    int hidden = static_cast<int>(x_.size(1));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    layernorm_half2_kernel<<<rows, threads, 0, stream>>>(
        reinterpret_cast<half*>(x_.data_ptr<at::Half>()),
        reinterpret_cast<half*>(gamma_.data_ptr<at::Half>()),
        reinterpret_cast<half*>(beta_.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        rows,
        hidden,
        static_cast<float>(eps));

    return y;
}

torch::Tensor launch_wrap(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    check_float_inputs(x, gamma, beta);
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

torch::Tensor launch_reduction(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    check_float_inputs(x, gamma, beta);
    auto x_ = x.contiguous();
    auto gamma_ = gamma.contiguous();
    auto beta_ = beta.contiguous();
    auto y = torch::empty_like(x_);

    int rows = static_cast<int>(x_.size(0));
    int hidden = static_cast<int>(x_.size(1));
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    layernorm_reduction_kernel<<<rows, hidden, 0, stream>>>(
        x_.data_ptr<float>(),
        gamma_.data_ptr<float>(),
        beta_.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        hidden,
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

torch::Tensor forward_reduction(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    return launch_reduction(x, gamma, beta, eps);
}

torch::Tensor forward_half2(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    return launch_half2(x, gamma, beta, eps);
}

torch::Tensor forward_welford(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    return launch_welford(x, gamma, beta, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_wrap", &forward_wrap, "Wrap warp-reduction layernorm CUDA kernel");
    m.def("forward_reduction", &forward_reduction, "Wrap reduction layernorm CUDA kernel");
    m.def("forward_half2", &forward_half2, "Wrap half2 layernorm CUDA kernel");
    m.def("forward_welford", &forward_welford, "Wrap welford layernorm CUDA kernel");
}
