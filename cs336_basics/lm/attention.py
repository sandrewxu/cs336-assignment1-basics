import token
from sympy import real_root
import torch
import math
from einops import einsum, rearrange
from cs336_basics.lm.rope import RotaryPositionalEmbedding

def softmax(x: torch.Tensor, i: int):
    """
    Apply softmax to the i-th dimension of input tensor x
    """
    x_max = torch.max(x, dim=i, keepdim=True).values
    x_mod = x - x_max
    exp_x = torch.exp(x_mod)
    x_sum = torch.sum(exp_x, dim=i, keepdim=True)
    result = exp_x / x_sum
    return result

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implement the scaled dot-product attention function.
    keys and queries of shape: (batch_size, ..., seq_len, d_k)
    values of shape: (batch_size, ..., seq_len, d_v)

    where ... represents any number of other batch_like dimensions

    return (batch-size, ..., d_v)

    support user-provided boolean mask of shape (seq_len, seq_len)
    """
    # calculate inside softmax
    d_k = K.size(-1)
    result = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    result = result / math.sqrt(d_k)

    # apply mask
    if mask is not None:
        result.masked_fill_(~mask, float('-inf'))

    # softmax, multiply by V
    result = softmax(result, -1)
    result = einsum(result, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return result

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None, use_rope: bool = False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.use_rope = use_rope

        if use_rope:
            assert max_seq_len is not None and theta is not None
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

        self.WO = torch.nn.Parameter(
            torch.empty((d_model, num_heads * self.d_v))
        )
        self.WQ = torch.nn.Parameter(
            torch.empty((num_heads * self.d_k, d_model))
        )
        self.WK = torch.nn.Parameter(
            torch.empty((num_heads * self.d_k, d_model))
        )
        self.WV = torch.nn.Parameter(
            torch.empty((num_heads * self.d_v, d_model))
        )
        std_dk = math.sqrt(2/(d_model+num_heads * self.d_k))
        std_dv = math.sqrt(2/(d_model+num_heads * self.d_v))
        torch.nn.init.trunc_normal_(self.WO, mean=0, std=std_dv, a=-3*std_dv, b=3*std_dv)
        torch.nn.init.trunc_normal_(self.WQ, mean=0, std=std_dk, a=-3*std_dk, b=3*std_dk)
        torch.nn.init.trunc_normal_(self.WK, mean=0, std=std_dk, a=-3*std_dk, b=3*std_dk)
        torch.nn.init.trunc_normal_(self.WV, mean=0, std=std_dv, a=-3*std_dv, b=3*std_dv)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, causal_mask: bool = True):
        """
        Apply MHSA to a tensor x (..., seq_len, d_model)

        returns
        (..., seq_len, d_model)
        """
        seq_len = x.size(-2)

        # product to Q, K, V
        Q = einsum(x, self.WQ, "... seq_len d_model, qkv d_model -> ... seq_len qkv")
        K = einsum(x, self.WK, "... seq_len d_model, qkv d_model -> ... seq_len qkv")
        V = einsum(x, self.WV, "... seq_len d_model, qkv d_model -> ... seq_len qkv")

        # reshape to separate heads
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = None
        if causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # apply attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # concatenate heads
        attn_output = rearrange(attn_output, "... h seq d -> ... seq (h d)")
        result = einsum(attn_output, self.WO, "... seq hd, d_model hd -> ... seq d_model")
        return result
