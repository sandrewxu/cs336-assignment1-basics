"""
Multi-head self-attention layer

Contains
- softmax
- scaled_dot_product_attention
- multihead_self_attention
"""

import math
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from typing import Optional

def softmax(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Apply softmax to the "i"-th dimension of the input tensor "x"
    Output tensor has same dimensions as input tensor
    """
    # subtract max of dim i from all elems of dim i
    x_norm = x - torch.max(x, dim=dim, keepdim=True).values
    # do softmax
    x_norm_exp = torch.exp(x_norm)
    softmax = x_norm_exp / torch.sum(x_norm_exp, dim=dim, keepdim=True)
    return softmax

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... queries d_k"],
    K: Float[torch.Tensor, "... keys d_k"],
    V: Float[torch.Tensor, "... values d_v"],
    mask: Optional[Bool[torch.Tensor, "... queries keys"]],
) -> Float[torch.Tensor, "... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, do
    the scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # Compute inner (QK^T / sqrt{d_k})
    inner = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    inner = inner / math.sqrt(Q.shape[-1])  # d_k
    if mask is not None:
        inner = inner.masked_fill(~mask, float('-inf'))
    softmax_inner = softmax(inner, dim=-1)  # normalize queries along keys
    return einsum(softmax_inner, V, "... queries keys, ... keys d_v -> ... queries d_v")

class multihead_self_attention(nn.Module):
    """
    multi_head_self_attention
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: Optional[float] = None,
        rope_max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # d_k = d_v = d_model / h
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        d_k = self.d_k
        d_v = self.d_k
        self.wqkv_weight = nn.Parameter(torch.empty((3 * num_heads * d_k, d_model), device=device, dtype=dtype))
        self.wo_weight = nn.Parameter(torch.empty((d_model, num_heads * d_v), device=device, dtype=dtype))

        qkv_std = math.sqrt(2 / (self.wqkv_weight.shape[0] + d_model))
        nn.init.trunc_normal_(self.wqkv_weight, mean=0, std=qkv_std, a=-3.0 * qkv_std, b=3.0 * qkv_std)
        o_std = math.sqrt(2 / (num_heads * d_v + d_model))
        nn.init.trunc_normal_(self.wo_weight, mean=0, std=o_std, a=-3.0 * o_std, b=3.0 * o_std)

        # Initialize RoPE if parameters are provided
        if rope_theta is not None and rope_max_seq_len is not None:
            from .rope import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=d_k,
                max_seq_len=rope_max_seq_len,
                device=device
            )
        else:
            self.rope = None

    def forward(
        self,
        x: Float[torch.Tensor, "... seq d_model"],
        token_positions: Optional[Int[torch.Tensor, "... seq"]] = None,
        causal_mask: Optional[bool] = True,
    ) -> Float[torch.Tensor, "... seq d_model"]:
        seq_len = x.shape[-2]
        # Project
        qkv = einsum(self.wqkv_weight, x, "qkhd_k d_model, ... seq d_model -> ... seq qkhd_k")
        qkv = rearrange(qkv, "... seq (qkv heads d_k) -> qkv ... heads seq d_k", qkv=3, heads=self.num_heads, d_k = self.d_k)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to queries and keys if available
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        # Create causal mask, broadcast to dims
        attn_mask = None
        if causal_mask:
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Compute attention, reshape back
        out = scaled_dot_product_attention(queries, keys, values, attn_mask) # [... h seq d_v]
        out = rearrange(out, "... heads seq d_v -> ... seq (heads d_v)")

        return einsum(self.wo_weight, out, "d_model hd_v, ... seq hd_v -> ... seq d_model")
