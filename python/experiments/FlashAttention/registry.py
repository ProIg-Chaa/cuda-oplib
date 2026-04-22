import pathlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
from torch.utils.cpp_extension import load

try:
    from refs import attention_python_ref, build_official_impl
except ImportError:
    from .refs import attention_python_ref, build_official_impl


THIS_DIR = pathlib.Path(__file__).resolve().parent
WRAP_SRC = THIS_DIR / "fsattention_wrap.cu"


@dataclass(frozen=True)
class FlashAttentionImpl:
    name: str
    stage: str
    supported_dtypes: tuple[str, ...]
    supported_devices: tuple[str, ...]
    notes: str
    builder: Callable[[torch.dtype, torch.device], Callable]


@lru_cache(maxsize=1)
def load_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    return load(
        name="flashattention_experiment_ext",
        sources=[str(WRAP_SRC)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )


def build_registry(device: torch.device):
    ext = load_extension() if device.type == "cuda" else None

    def official_builder(dtype: torch.dtype, run_device: torch.device):
        return build_official_impl(dtype, run_device)

    def python_ref_builder(dtype: torch.dtype, run_device: torch.device):
        del dtype, run_device
        return attention_python_ref

    def ext_builder(method_name: str):
        def builder(dtype: torch.dtype, run_device: torch.device):
            if ext is None:
                raise RuntimeError("CUDA extension is only available on CUDA")
            del dtype, run_device
            method = getattr(ext, method_name)

            def forward(q, k, v):
                return method(q, k, v)

            return forward

        return builder

    return [
        FlashAttentionImpl(
            name="torch_official",
            stage="baseline",
            supported_dtypes=("float32",),
            supported_devices=("cpu", "cuda"),
            notes="torch scaled_dot_product_attention reference",
            builder=official_builder,
        ),
        FlashAttentionImpl(
            name="torch_python",
            stage="baseline",
            supported_dtypes=("float32",),
            supported_devices=("cpu", "cuda"),
            notes="manual attention reference",
            builder=python_ref_builder,
        ),
        FlashAttentionImpl(
            name="online_warp",
            stage="candidate",
            supported_dtypes=("float32",),
            supported_devices=("cuda",),
            notes="fsattention online warp CUDA kernel",
            builder=ext_builder("forward_online_warp_f32"),
        ),
    ]
