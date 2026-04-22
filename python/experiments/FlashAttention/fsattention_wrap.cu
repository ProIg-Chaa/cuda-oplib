#include <torch/extension.h>

#include "fsattention.cu"

namespace {

constexpr int kBlockSize = 128;
constexpr int kDebugBlockSize = 32;
constexpr int kMaxFrag = 8;

void check_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.dtype() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "q, k, v must be 2D tensors");
    TORCH_CHECK(q.size(1) == k.size(1), "q and k hidden sizes must match");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v sequence lengths must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v hidden sizes must match");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, v must be contiguous");
    TORCH_CHECK(
        ((q.size(1) + 31) / 32) <= kMaxFrag,
        "hidden dimension exceeds kMaxFrag capacity");
}

torch::Tensor forward_online_warp_f32(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    check_inputs(q, k, v);

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();
    auto out = torch::zeros({q.size(0), q.size(1)}, q.options());

    dim3 block(kBlockSize);
    dim3 grid(q.size(0));
    onlinesfatt_forward_f32_warp_kernel<kBlockSize, kMaxFrag>
        <<<grid, block>>>(
            q_contig.data_ptr<float>(),
            k_contig.data_ptr<float>(),
            v_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            static_cast<int>(q.size(0)),
            static_cast<int>(k.size(0)),
            static_cast<int>(q.size(1)),
            0);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "FlashAttention kernel launch failed");
    return out;
}

torch::Tensor forward_online_warp_f32_debug(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    check_inputs(q, k, v);

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();
    auto out = torch::zeros({q.size(0), q.size(1)}, q.options());

    dim3 block(kDebugBlockSize);
    dim3 grid(q.size(0));
    onlinesfatt_forward_f32_warp_kernel<kDebugBlockSize, kMaxFrag>
        <<<grid, block>>>(
            q_contig.data_ptr<float>(),
            k_contig.data_ptr<float>(),
            v_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            static_cast<int>(q.size(0)),
            static_cast<int>(k.size(0)),
            static_cast<int>(q.size(1)),
            1);
    TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "FlashAttention debug sync failed");
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "FlashAttention kernel launch failed");
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_online_warp_f32", &forward_online_warp_f32, "FlashAttention online warp f32");
    m.def("forward_online_warp_f32_debug", &forward_online_warp_f32_debug, "FlashAttention online warp f32 debug");
}
