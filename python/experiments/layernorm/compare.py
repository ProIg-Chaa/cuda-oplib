import argparse

import torch

try:
    from cases import CASES
    from registry import build_registry
    from report import render_markdown_table, render_terminal_table
except ImportError:
    from .cases import CASES
    from .registry import build_registry
    from .report import render_markdown_table, render_terminal_table


def benchmark_cuda(fn, warmup: int, iters: int) -> float:
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


def benchmark_cpu(fn, warmup: int, iters: int) -> float:
    import time

    for _ in range(warmup):
        fn()

    begin = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - begin) * 1000.0 / iters


def parse_args():
    parser = argparse.ArgumentParser(description="Unified LayerNorm experiment driver")
    parser.add_argument("--case", choices=sorted(CASES.keys()), default="main_fp32")
    parser.add_argument(
        "--impls",
        default="all",
        help="Comma-separated implementation names to run. Default: all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a markdown table after the terminal table",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    case = CASES[args.case]
    dtype_name = case["dtype"]
    dtype = getattr(torch, dtype_name)
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(args.seed)
    rows = case["rows"]
    hidden = case["hidden"]
    eps = case["eps"]
    warmup = case["warmup"]
    iters = case["iters"]

    x = torch.randn(rows, hidden, device=device, dtype=dtype)
    gamma = torch.randn(hidden, device=device, dtype=dtype)
    beta = torch.randn(hidden, device=device, dtype=dtype)

    registry = build_registry(hidden, device)
    if args.impls != "all":
        selected = {name.strip() for name in args.impls.split(",") if name.strip()}
        registry = [impl for impl in registry if impl.name in selected]

    baseline_impl = next(impl for impl in registry if impl.name == "torch_official")
    baseline_fn = baseline_impl.builder(dtype, device)
    with torch.no_grad():
        y_ref = baseline_fn(x, gamma, beta, eps)

    if device.type == "cuda":
        baseline_ms = benchmark_cuda(lambda: baseline_fn(x, gamma, beta, eps), warmup, iters)
    else:
        baseline_ms = benchmark_cpu(lambda: baseline_fn(x, gamma, beta, eps), warmup, iters)

    rows_out = []
    for impl in registry:
        if dtype_name not in impl.supported_dtypes:
            rows_out.append(
                {
                    "name": impl.name,
                    "stage": impl.stage,
                    "dtype": dtype_name,
                    "correct": "skip",
                    "max_abs_diff": "--",
                    "avg_ms": "--",
                    "speedup_vs_ref": "--",
                    "notes": f"unsupported for {dtype_name}",
                }
            )
            continue

        fn = impl.builder(dtype, device)
        with torch.no_grad():
            y = fn(x, gamma, beta, eps)

        max_abs_diff = (y.float() - y_ref.float()).abs().max().item()
        allclose = torch.allclose(y.float(), y_ref.float(), atol=1e-3, rtol=1e-3)

        if not allclose:
            rows_out.append(
                {
                    "name": impl.name,
                    "stage": impl.stage,
                    "dtype": dtype_name,
                    "correct": "no",
                    "max_abs_diff": f"{max_abs_diff:.3e}",
                    "avg_ms": "--",
                    "speedup_vs_ref": "--",
                    "notes": impl.notes,
                }
            )
            continue

        if device.type == "cuda":
            avg_ms = benchmark_cuda(lambda: fn(x, gamma, beta, eps), warmup, iters)
        else:
            avg_ms = benchmark_cpu(lambda: fn(x, gamma, beta, eps), warmup, iters)

        speedup = baseline_ms / avg_ms if avg_ms > 0 else float("inf")
        rows_out.append(
            {
                "name": impl.name,
                "stage": impl.stage,
                "dtype": dtype_name,
                "correct": "yes",
                "max_abs_diff": f"{max_abs_diff:.3e}",
                "avg_ms": f"{avg_ms:.3f}",
                "speedup_vs_ref": f"{speedup:.3f}",
                "notes": impl.notes,
            }
        )

    print(
        f"case={args.case} rows={rows} hidden={hidden} dtype={dtype_name} "
        f"device={device} warmup={warmup} iters={iters}"
    )
    print(render_terminal_table(rows_out))

    if args.markdown:
        print()
        print(render_markdown_table(rows_out))


if __name__ == "__main__":
    main()

