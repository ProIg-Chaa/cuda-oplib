import pathlib

import torch
from torch.utils.cpp_extension import load


THIS_DIR = pathlib.Path(__file__).resolve().parent
WRAP_SRC = THIS_DIR / "layernorm_wrap.cu"


def bench_cuda(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    ext = load(
        name="layernorm_min_ext",
        sources=[str(WRAP_SRC)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )

    batch = 512
    hidden = 768
    eps = 1e-5
    dtype = torch.float32
    device = "cuda"

    torch.manual_seed(0)
    x = torch.randn(batch, hidden, device=device, dtype=dtype)
    gamma = torch.randn(hidden, device=device, dtype=dtype)
    beta = torch.randn(hidden, device=device, dtype=dtype)

    ln = torch.nn.LayerNorm(hidden, eps=eps, elementwise_affine=True).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        ln.weight.copy_(gamma)
        ln.bias.copy_(beta)

    official_ms = bench_cuda(lambda: ln(x))
    wrap_ms = bench_cuda(lambda: ext.forward_wrap(x, gamma, beta, eps))
    reduction_ms = bench_cuda(lambda: ext.forward_reduction(x, gamma, beta, eps))
    welford_ms = bench_cuda(lambda: ext.forward_welford(x, gamma, beta, eps))

    print(f"B={batch}, D={hidden}, device={device}, dtype={dtype}")
    print(f"official torch.nn.LayerNorm : {official_ms:.3f} ms / iter")
    print(f"warp cuda kernel            : {wrap_ms:.3f} ms / iter")
    print(f"reduction cuda kernel       : {reduction_ms:.3f} ms / iter")
    print(f"welford cuda kernel         : {welford_ms:.3f} ms / iter")
    print(f"warp / official             : {wrap_ms / official_ms:.3f}x slower")
    print(f"reduction / official        : {reduction_ms / official_ms:.3f}x slower")
    print(f"welford / official          : {welford_ms / official_ms:.3f}x slower")


if __name__ == "__main__":
    main()
