import pathlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
from torch.utils.cpp_extension import load

try:
    from refs import add_then_rmsnorm_with_affine, build_official_impl, rmsnorm_with_affine
except ImportError:
    from .refs import add_then_rmsnorm_with_affine, build_official_impl, rmsnorm_with_affine


THIS_DIR = pathlib.Path(__file__).resolve().parent
WRAP_SRC = THIS_DIR / "rmsnorm_wrap.cu"


@dataclass(frozen=True)
class RMSNormImpl:
    name: str
    stage: str
    supported_dtypes: tuple[str, ...]
    notes: str
    builder: Callable[[torch.dtype, torch.device], Callable]


@lru_cache(maxsize=1)
def load_extension():
    return load(
        name="rmsnorm_experiment_ext",
        sources=[str(WRAP_SRC)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )


def build_registry(hidden: int, device: torch.device):
    ext = load_extension() if device.type == "cuda" else None

    def official_builder(dtype: torch.dtype, run_device: torch.device):
        return build_official_impl(hidden, dtype, run_device)

    def python_ref_builder(dtype: torch.dtype, run_device: torch.device):
        del dtype, run_device

        def forward(x, gamma, eps, residual=None):
            if residual is not None:
                return add_then_rmsnorm_with_affine(x, residual, gamma, eps)
            return rmsnorm_with_affine(x, gamma, eps)

        return forward

    def ext_builder(method_name: str):
        def builder(dtype: torch.dtype, run_device: torch.device):
            if ext is None:
                raise RuntimeError("CUDA extension is only available on CUDA")
            del dtype, run_device
            method = getattr(ext, method_name)

            def forward(x, gamma, eps, residual=None):
                if residual is not None:
                    return method(x + residual, gamma, eps)
                return method(x, gamma, eps)

            return forward

        return builder

    def fused_ext_builder(method_name: str):
        def builder(dtype: torch.dtype, run_device: torch.device):
            if ext is None:
                raise RuntimeError("CUDA extension is only available on CUDA")
            del dtype, run_device
            method = getattr(ext, method_name)

            def forward(x, gamma, eps, residual=None):
                if residual is None:
                    raise RuntimeError("fused add + rmsnorm implementation requires residual")
                return method(x, gamma, residual, eps)

            return forward

        return builder

    return [
        RMSNormImpl(
            name="torch_official",
            stage="baseline",
            supported_dtypes=("float32", "float16"),
            notes="torch.nn.RMSNorm reference if available",
            builder=official_builder,
        ),
        RMSNormImpl(
            name="torch_python",
            stage="baseline",
            supported_dtypes=("float32", "float16"),
            notes="manual PyTorch tensor reference",
            builder=python_ref_builder,
        ),
        RMSNormImpl(
            name="f32_warp",
            stage="candidate",
            supported_dtypes=("float32",),
            notes="f32 warp-reduction CUDA kernel",
            builder=ext_builder("forward_f32"),
        ),
        RMSNormImpl(
            name="half2_warp",
            stage="candidate",
            supported_dtypes=("float16",),
            notes="half2 warp-reduction CUDA kernel",
            builder=ext_builder("forward_half2"),
        ),
        RMSNormImpl(
            name="fused_add_f32",
            stage="candidate",
            supported_dtypes=("float32",),
            notes="fused add + f32 warp-reduction CUDA kernel",
            builder=fused_ext_builder("forward_fused_add_f32"),
        ),
        RMSNormImpl(
            name="fused_add_half2",
            stage="candidate",
            supported_dtypes=("float16",),
            notes="fused add + half2 warp-reduction CUDA kernel",
            builder=fused_ext_builder("forward_fused_add_half2"),
        ),
    ]
