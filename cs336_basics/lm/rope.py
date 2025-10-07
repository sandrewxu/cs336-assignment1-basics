import torch
from einops import einsum, repeat

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even (pairs of dims are rotated)."
        self.theta = theta
        self.d_k = d_k
        self.max_seq_length = max_seq_len

        # we need d/2 rotation matrices
        ks = torch.arange(d_k//2, dtype = torch.float32, device=device) # [d_k/2]
        scales = self.theta ** (-2 * ks/d_k) # [d_k/2]

        # positions
        positions = torch.arange(max_seq_len, dtype = torch.float32, device=device) # [max_seq_len]

        # phases = positions[i] * scales[k]
        phases = einsum(positions, scales, "max_seq_len, half_d -> max_seq_len half_d")
        phases_full = repeat(phases, 'n d -> n (d repeat)', repeat=2)

        sin = torch.sin(phases_full)
        cos = torch.cos(phases_full)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        """
        # index sin and cos along the correct token positions
        sin_pos = self.sin[token_positions.to(torch.long)].to(x.device, x.dtype)
        cos_pos = self.cos[token_positions.to(torch.long)].to(x.device, x.dtype)

        # split even/odd channels
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        s = sin_pos[..., ::2]
        c = cos_pos[..., ::2]

        # compute the 2x2 @ 2x1 matrix multiplication for each element
        out_even = x_even * c - x_odd * s
        out_odd = x_even * s + x_odd * c

        out = torch.stack((out_even, out_odd), dim=-1).reshape(x.shape)
        return out
