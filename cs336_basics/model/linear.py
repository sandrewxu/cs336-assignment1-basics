"""
Linear layer

Used in:
- Transformer block (position-wise feed-forward)
- Output embedding layer
"""

import math

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

class Linear(nn.Module):
    """
    Linear layer without bias

    Transforms in_features into out_features: y = xW^T
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_out"]:
        """
        Apply linear transformation: y = xW^T
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
