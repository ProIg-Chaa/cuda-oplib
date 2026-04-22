import math

import torch
import torch.nn.functional as F


def attention_python_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    scores = (q @ k.transpose(0, 1)) / math.sqrt(q.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def build_official_impl(dtype: torch.dtype, device: torch.device):
    del dtype, device

    def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q4 = q.unsqueeze(0).unsqueeze(0)
        k4 = k.unsqueeze(0).unsqueeze(0)
        v4 = v.unsqueeze(0).unsqueeze(0)
        out = F.scaled_dot_product_attention(q4, k4, v4, dropout_p=0.0)
        return out.squeeze(0).squeeze(0)

    return forward
