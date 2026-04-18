import torch


def rmsnorm_with_affine(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    rms = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return x * rms * gamma


def build_official_impl(hidden: int, dtype: torch.dtype, device: torch.device):
    if hasattr(torch.nn, "RMSNorm"):
        layer = torch.nn.RMSNorm(hidden, eps=1e-5, elementwise_affine=True).to(
            device=device, dtype=dtype
        )

        def forward(x: torch.Tensor, gamma: torch.Tensor, eps: float):
            layer.eps = eps
            with torch.no_grad():
                layer.weight.copy_(gamma)
            return layer(x)

        return forward

    def fallback_forward(x: torch.Tensor, gamma: torch.Tensor, eps: float):
        return rmsnorm_with_affine(x, gamma, eps)

    return fallback_forward

