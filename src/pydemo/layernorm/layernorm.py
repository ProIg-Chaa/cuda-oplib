import argparse
import pathlib
import time

import torch
from torch.utils.cpp_extension import load


THIS_DIR = pathlib.Path(__file__).resolve().parent
CUDA_SRC = THIS_DIR / "layernorm.cu"


def layernorm_with_affine(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return x_hat * gamma + beta


def load_cuda_extension():
    return load(
        name="layernorm_naive_ext",
        sources=[str(CUDA_SRC)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )


def benchmark_cuda(fn, warmup, iters):
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


def benchmark_cpu(fn, warmup, iters):
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please use --device cpu")

    x = torch.randn(args.batch, args.hidden, dtype=dtype, device=device)
    ln = torch.nn.LayerNorm(args.hidden, eps=args.eps, elementwise_affine=True).to(
        device=device, dtype=dtype
    )

    with torch.no_grad():
        gamma = ln.weight.detach().clone()
        beta = ln.bias.detach().clone()
        y_official = ln(x)
        y_python = layernorm_with_affine(x, gamma, beta, eps=args.eps)

    print("Correctness check:")
    print(
        f"python vs official: max_abs_diff="
        f"{(y_official - y_python).abs().max().item():.6g}, "
        f"allclose={torch.allclose(y_official, y_python, atol=1e-4, rtol=1e-4)}"
    )

    if device.type == "cuda":
        ext = load_cuda_extension()
        x_cuda = x.float().contiguous()
        gamma_cuda = gamma.float().contiguous()
        beta_cuda = beta.float().contiguous()

        with torch.no_grad():
            y_cuda = ext.forward(x_cuda, gamma_cuda, beta_cuda, args.eps)

        print(
            f"cuda kernel vs official: max_abs_diff="
            f"{(y_official.float() - y_cuda).abs().max().item():.6g}, "
            f"allclose={torch.allclose(y_official.float(), y_cuda, atol=1e-4, rtol=1e-4)}"
        )

        official_ms = benchmark_cuda(lambda: ln(x), args.warmup, args.iters)
        python_ms = benchmark_cuda(
            lambda: layernorm_with_affine(x, gamma, beta, args.eps),
            args.warmup,
            args.iters,
        )
        cuda_ms = benchmark_cuda(
            lambda: ext.forward(x_cuda, gamma_cuda, beta_cuda, args.eps),
            args.warmup,
            args.iters,
        )

        print("-" * 72)
        print(
            f"B={args.batch}, D={args.hidden}, device={device}, dtype={dtype}, "
            f"note=naive cuda kernel requires hidden <= 1024"
        )
        print(f"official torch.nn.LayerNorm : {official_ms:.3f} ms / iter")
        print(f"python tensor implementation : {python_ms:.3f} ms / iter")
        print(f"current cuda naive kernel    : {cuda_ms:.3f} ms / iter")
        print(f"official / python            : {official_ms / python_ms:.3f}x")
        print(f"official / cuda              : {official_ms / cuda_ms:.3f}x")
        print(f"python / cuda                : {python_ms / cuda_ms:.3f}x")
    else:
        official_ms = benchmark_cpu(lambda: ln(x), args.warmup, args.iters)
        python_ms = benchmark_cpu(
            lambda: layernorm_with_affine(x, gamma, beta, args.eps),
            args.warmup,
            args.iters,
        )
        print("-" * 72)
        print(f"B={args.batch}, D={args.hidden}, device={device}, dtype={dtype}")
        print(f"official torch.nn.LayerNorm : {official_ms:.3f} ms / iter")
        print(f"python tensor implementation : {python_ms:.3f} ms / iter")


if __name__ == "__main__":
    main()
