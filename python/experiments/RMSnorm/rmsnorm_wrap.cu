#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define main rmsnorm_demo_main
#include "rmsnorm.cu"
#undef main

#define main fused_add_rmsnorm_demo_main
#include "fused_add_rmsnorm.cu"
#undef main

namespace {

void check_float_inputs(const torch::Tensor& x, const torch::Tensor& gamma) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(x.size(1) == gamma.size(0), "hidden dim must match gamma");
}

void check_float_inputs_with_residual(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& residual) {
    check_float_inputs(x, gamma);
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x");
}

void check_half_inputs(const torch::Tensor& x, const torch::Tensor& gamma) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat16, "gamma must be float16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(x.size(1) == gamma.size(0), "hidden dim must match gamma");
}

void check_half_inputs_with_residual(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& residual) {
    check_half_inputs(x, gamma);
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "residual must be float16");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x");
}

torch::Tensor launch_f32(torch::Tensor x, torch::Tensor gamma, double eps) {
    check_float_inputs(x, gamma);
    auto x_ = x.contiguous();
    auto gamma_ = gamma.contiguous();
    auto y = torch::empty_like(x_);

    int rows = static_cast<int>(x_.size(0));
    int hidden = static_cast<int>(x_.size(1));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    rmsnorm_forward_f32_warp_kernel<threads><<<rows, threads, 0, stream>>>(
        x_.data_ptr<float>(),
        gamma_.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        hidden,
        static_cast<float>(eps));

    return y;
}

torch::Tensor launch_fused_add_f32(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor residual,
    double eps) {
    check_float_inputs_with_residual(x, gamma, residual);
    auto x_ = x.contiguous();
    auto gamma_ = gamma.contiguous();
    auto residual_ = residual.contiguous();
    auto y = torch::empty_like(x_);

    int rows = static_cast<int>(x_.size(0));
    int hidden = static_cast<int>(x_.size(1));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    fused_add_rmsnorm_forward_f32_warp_kernel<threads><<<rows, threads, 0, stream>>>(
        x_.data_ptr<float>(),
        gamma_.data_ptr<float>(),
        residual_.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        hidden,
        static_cast<float>(eps));

    return y;
}

torch::Tensor launch_half2(torch::Tensor x, torch::Tensor gamma, double eps) {
    check_half_inputs(x, gamma);
    auto x_ = x.contiguous();
    auto gamma_ = gamma.contiguous();
    auto y = torch::empty_like(x_);

    int rows = static_cast<int>(x_.size(0));
    int hidden = static_cast<int>(x_.size(1));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    rmsnorm_forward_h2_warp_kernel<threads><<<rows, threads, 0, stream>>>(
        reinterpret_cast<const half*>(x_.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gamma_.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        rows,
        hidden,
        hidden,
        static_cast<float>(eps));

    return y;
}

torch::Tensor launch_fused_add_half2(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor residual,
    double eps) {
    check_half_inputs_with_residual(x, gamma, residual);
    auto x_ = x.contiguous();
    auto gamma_ = gamma.contiguous();
    auto residual_ = residual.contiguous();
    auto y = torch::empty_like(x_);

    int rows = static_cast<int>(x_.size(0));
    int hidden = static_cast<int>(x_.size(1));
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    fused_add_rmsnorm_forward_h2_warp<threads><<<rows, threads, 0, stream>>>(
        reinterpret_cast<const half*>(x_.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gamma_.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual_.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        rows,
        hidden,
        static_cast<float>(eps));

    return y;
}

}  // namespace

torch::Tensor forward_f32(torch::Tensor x, torch::Tensor gamma, double eps) {
    return launch_f32(x, gamma, eps);
}

torch::Tensor forward_half2(torch::Tensor x, torch::Tensor gamma, double eps) {
    return launch_half2(x, gamma, eps);
}

torch::Tensor forward_fused_add_f32(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor residual,
    double eps) {
    return launch_fused_add_f32(x, gamma, residual, eps);
}

torch::Tensor forward_fused_add_half2(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor residual,
    double eps) {
    return launch_fused_add_half2(x, gamma, residual, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_f32", &forward_f32, "Wrap f32 RMSNorm CUDA kernel");
    m.def("forward_half2", &forward_half2, "Wrap half2 RMSNorm CUDA kernel");
    m.def(
        "forward_fused_add_f32",
        &forward_fused_add_f32,
        "Wrap fused add + f32 RMSNorm CUDA kernel");
    m.def(
        "forward_fused_add_half2",
        &forward_fused_add_half2,
        "Wrap fused add + half2 RMSNorm CUDA kernel");
}
