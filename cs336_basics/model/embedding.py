"""
Embedding layer

Used in:
- Token embedding layer
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int

class Embedding(nn.Module):
    """
    Embedding layer

    Creates embedding layer.
    Forward method indexes embedding matrix of shape (vocab_size, d_model)
    uses torch.LongTensor of token IDs with shape (batch_size, sequence_length)
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(
        self,
        token_ids: Int[torch.LongTensor, "..."]
    ) -> Float[torch.Tensor, "... d_model"]:
        """
        Look up embeddings for token ids.
        """
        return self.weight[token_ids]
