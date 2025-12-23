"""
Transformer Block and LM
"""

from jaxtyping import Int, Float
import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadSelfAttention
from .embedding import Embedding
from .feedforward import SwiGLU
from .linear import Linear
from .normalization import RMSNorm

class TransformerBlock(nn.Module):
    """
    Transformer block
    
    Input x: (b, s, d_model)
    y = x + MultiHeadSelfAttention(RMSNorm(x))
    z = y + FeedForward(RMSNorm(y))
    Output x: (b, s, d_model)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: Optional[float],
        rope_max_seq_len: Optional[int],
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_theta, rope_max_seq_len)
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.attn_norm(x))
        z = y + self.ff(self.ff_norm(y))
        return z

class TransformerLM(nn.Module):
    """
    Transformer LM
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope_theta,
                context_length,
            ) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lin_output = Linear(d_model, vocab_size)

    def forward(self, x: Float[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = self.final_norm(x)
        x = self.lin_output(x)
        return x
