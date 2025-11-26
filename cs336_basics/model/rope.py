"""
RoPE layer

Used in:
- attention
"""

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float, Int

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings

    given an input tensor, apply a rotation matrix given token positions
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0
        # compute inv freqs: 1 / (theta^(2i/d_k))
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        # go across all possible positions (1 to seq_len)
        positions = torch.arange(max_seq_len, device=device)
        angles = einsum(positions, inv_freq, "max_seq_len, d -> max_seq_len d")
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply rotary positional embeddings to input tensor
        """
        # Get position-specific rotation angles
        cos_theta = self.cos[token_positions]
        sin_theta = self.sin[token_positions]

        # split into pairs for rotation
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply 2D rotation: [cos -sin; sin cos]
        out_even = x_even * cos_theta - x_odd * sin_theta
        out_odd = x_even * sin_theta + x_odd * cos_theta

        # Interleave back to original layout
        out = torch.stack((out_even, out_odd), dim=-1).reshape(x.shape)
        return out
