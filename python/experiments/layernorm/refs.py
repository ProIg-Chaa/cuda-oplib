import torch


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


def build_official_impl(hidden: int, dtype: torch.dtype, device: torch.device):
    layer = torch.nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=True).to(
        device=device, dtype=dtype
    )

    def forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float):
        layer.eps = eps
        with torch.no_grad():
            layer.weight.copy_(gamma)
            layer.bias.copy_(beta)
        return layer(x)

    return forward

