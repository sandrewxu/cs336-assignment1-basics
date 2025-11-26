"""
Normalization layer

Currently implemented RMSNorm, normalizing each individual vector of size (d_model)

Used in:
- post-Transformer
- in Transformer
"""

import torch
import torch.nn as nn
from jaxtyping import Float

class RMSNorm(nn.Module):
    """
    RMSNorm layer without bias

    Normalizes tensors on d_model
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones((self.d_model), device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        assert x.size(-1) == self.d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_normed = x / self._rms(x)
        result = x_normed * self.weight

        return result.to(in_dtype)

    def _rms(
        self,
        x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... 1"]:
        # keepdim keeps the last dimension but collapses the number to 1
        return torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
