"""
Feed-forward network layer

Used in:
- Transformer block
"""

import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.model import Linear

def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network

    SiLU activation function and GLU

    d_in is the input embedding
    d_ff is the hidden layer embedding
    """
    def __init__(
        self,
        d_in: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_in, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_in, device=device, dtype=dtype)
        self.w3 = Linear(d_in, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_in"]:
        term1 = silu(self.w1(x))
        term2 = self.w3(x)
        combined = term1 * term2
        result = self.w2(combined)
        return result
