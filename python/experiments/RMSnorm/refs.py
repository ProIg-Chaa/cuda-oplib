import torch


def rmsnorm_with_affine(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    rms = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return x * rms * gamma


def add_then_rmsnorm_with_affine(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return rmsnorm_with_affine(x + residual, gamma, eps)


def build_official_impl(hidden: int, dtype: torch.dtype, device: torch.device):
    if hasattr(torch.nn, "RMSNorm"):
        layer = torch.nn.RMSNorm(hidden, eps=1e-5, elementwise_affine=True).to(
            device=device, dtype=dtype
        )

        def forward(
            x: torch.Tensor,
            gamma: torch.Tensor,
            eps: float,
            residual: torch.Tensor | None = None,
        ):
            layer.eps = eps
            with torch.no_grad():
                layer.weight.copy_(gamma)
            if residual is not None:
                x = x + residual
            return layer(x)

        return forward

    def fallback_forward(
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None = None,
    ):
        if residual is not None:
            return add_then_rmsnorm_with_affine(x, residual, gamma, eps)
        return rmsnorm_with_affine(x, gamma, eps)

    return fallback_forward
