import pathlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
from torch.utils.cpp_extension import load

try:
    from refs import build_official_impl, layernorm_with_affine
except ImportError:
    from .refs import build_official_impl, layernorm_with_affine


THIS_DIR = pathlib.Path(__file__).resolve().parent
WRAP_SRC = THIS_DIR / "layernorm_wrap.cu"


@dataclass(frozen=True)
class LayerNormImpl:
    name: str
    stage: str
    supported_dtypes: tuple[str, ...]
    notes: str
    builder: Callable[[torch.dtype, torch.device], Callable]


@lru_cache(maxsize=1)
def load_extension():
    return load(
        name="layernorm_experiment_ext",
        sources=[str(WRAP_SRC)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )


def build_registry(hidden: int, device: torch.device):
    ext = load_extension() if device.type == "cuda" else None

    def official_builder(dtype: torch.dtype, run_device: torch.device):
        return build_official_impl(hidden, dtype, run_device)

    def python_ref_builder(dtype: torch.dtype, run_device: torch.device):
        del run_device

        def forward(x, gamma, beta, eps):
            return layernorm_with_affine(x, gamma, beta, eps)

        return forward

    def ext_builder(method_name: str):
        def builder(dtype: torch.dtype, run_device: torch.device):
            if ext is None:
                raise RuntimeError("CUDA extension is only available on CUDA")
            del dtype, run_device
            method = getattr(ext, method_name)

            def forward(x, gamma, beta, eps):
                return method(x, gamma, beta, eps)

            return forward

        return builder

    return [
        LayerNormImpl(
            name="torch_official",
            stage="baseline",
            supported_dtypes=("float32", "float16"),
            notes="torch.nn.LayerNorm reference",
            builder=official_builder,
        ),
        LayerNormImpl(
            name="torch_python",
            stage="baseline",
            supported_dtypes=("float32", "float16"),
            notes="manual PyTorch tensor reference",
            builder=python_ref_builder,
        ),
        LayerNormImpl(
            name="warp",
            stage="candidate",
            supported_dtypes=("float32",),
            notes="warp-reduction CUDA kernel",
            builder=ext_builder("forward_wrap"),
        ),
        LayerNormImpl(
            name="reduction",
            stage="draft",
            supported_dtypes=("float32",),
            notes="shared-memory reduction CUDA kernel",
            builder=ext_builder("forward_reduction"),
        ),
        LayerNormImpl(
            name="welford",
            stage="candidate",
            supported_dtypes=("float32",),
            notes="Welford CUDA kernel",
            builder=ext_builder("forward_welford"),
        ),
        LayerNormImpl(
            name="half2",
            stage="candidate",
            supported_dtypes=("float16",),
            notes="half2 CUDA kernel with float accumulation",
            builder=ext_builder("forward_half2"),
        ),
    ]
