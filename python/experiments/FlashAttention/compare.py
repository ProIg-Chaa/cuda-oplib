import argparse
from datetime import datetime
from pathlib import Path

import torch

try:
    from cases import CASES
    from registry import build_registry
    from report import render_markdown_table, render_terminal_table
except ImportError:
    from .cases import CASES
    from .registry import build_registry
    from .report import render_markdown_table, render_terminal_table


THIS_DIR = Path(__file__).resolve().parent
EXPLOG_DIR = THIS_DIR.parent / "explog"


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
    parser = argparse.ArgumentParser(description="Unified FlashAttention experiment driver")
    parser.add_argument("--case", choices=sorted(CASES.keys()), default="debug_fp32")
    parser.add_argument("--impls", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    return parser.parse_args()


def build_log_markdown(
    *,
    case_name: str,
    sq: int,
    sk: int,
    hidden: int,
    dtype_name: str,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
    rows_out: list[dict],
    timestamp: datetime,
) -> str:
    header_lines = [
        "# FlashAttention Experiment Log",
        "",
        f"- Date: `{timestamp.isoformat(timespec='seconds')}`",
        f"- Case: `{case_name}`",
        f"- Sq: `{sq}`",
        f"- Sk: `{sk}`",
        f"- Hidden: `{hidden}`",
        f"- Dtype: `{dtype_name}`",
        f"- Device: `{device}`",
        f"- Warmup: `{warmup}`",
        f"- Iters: `{iters}`",
        f"- Seed: `{seed}`",
        "",
        "## Results",
        "",
        render_markdown_table(rows_out),
        "",
    ]
    return "\n".join(header_lines)


def write_experiment_log(
    *,
    case_name: str,
    sq: int,
    sk: int,
    hidden: int,
    dtype_name: str,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
    rows_out: list[dict],
) -> Path:
    timestamp = datetime.now()
    dated_dir = EXPLOG_DIR / "FlashAttention" / timestamp.strftime("%Y-%m-%d")
    dated_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{timestamp.strftime('%H%M%S')}_{case_name}_{dtype_name}_{device.type}.md"
    log_path = dated_dir / filename
    log_path.write_text(
        build_log_markdown(
            case_name=case_name,
            sq=sq,
            sk=sk,
            hidden=hidden,
            dtype_name=dtype_name,
            device=device,
            warmup=warmup,
            iters=iters,
            seed=seed,
            rows_out=rows_out,
            timestamp=timestamp,
        ),
        encoding="utf-8",
    )
    return log_path


def main():
    args = parse_args()
    case = CASES[args.case]
    dtype_name = case["dtype"]
    dtype = getattr(torch, dtype_name)
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(args.seed)
    sq = case["sq"]
    sk = case["sk"]
    hidden = case["hidden"]
    warmup = case["warmup"]
    iters = case["iters"]

    q = torch.randn(sq, hidden, device=device, dtype=dtype)
    k = torch.randn(sk, hidden, device=device, dtype=dtype)
    v = torch.randn(sk, hidden, device=device, dtype=dtype)

    registry = build_registry(device)
    if args.impls != "all":
        selected = {name.strip() for name in args.impls.split(",") if name.strip()}
        registry = [impl for impl in registry if impl.name in selected]

    baseline_impl = next(impl for impl in registry if impl.name == "torch_official")
    baseline_fn = baseline_impl.builder(dtype, device)
    with torch.no_grad():
        y_ref = baseline_fn(q, k, v)

    if device.type == "cuda":
        baseline_ms = benchmark_cuda(lambda: baseline_fn(q, k, v), warmup, iters)
    else:
        baseline_ms = benchmark_cpu(lambda: baseline_fn(q, k, v), warmup, iters)

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

        if device.type not in impl.supported_devices:
            rows_out.append(
                {
                    "name": impl.name,
                    "stage": impl.stage,
                    "dtype": dtype_name,
                    "correct": "skip",
                    "max_abs_diff": "--",
                    "avg_ms": "--",
                    "speedup_vs_ref": "--",
                    "notes": f"unsupported on {device.type}",
                }
            )
            continue

        fn = impl.builder(dtype, device)
        with torch.no_grad():
            y = fn(q, k, v)

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
            avg_ms = benchmark_cuda(lambda: fn(q, k, v), warmup, iters)
        else:
            avg_ms = benchmark_cpu(lambda: fn(q, k, v), warmup, iters)

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
        f"case={args.case} sq={sq} sk={sk} hidden={hidden} "
        f"dtype={dtype_name} device={device.type} warmup={warmup} iters={iters}"
    )
    print(render_terminal_table(rows_out))

    if args.markdown:
        print()
        print(render_markdown_table(rows_out))

    if not args.no_log:
        log_path = write_experiment_log(
            case_name=args.case,
            sq=sq,
            sk=sk,
            hidden=hidden,
            dtype_name=dtype_name,
            device=device,
            warmup=warmup,
            iters=iters,
            seed=args.seed,
            rows_out=rows_out,
        )
        print()
        print(f"saved markdown log: {log_path}")


if __name__ == "__main__":
    main()
