"""
Core language model

Goal: take a batched sequence of integer token IDs (i.e., torch.Tensor of shape 
(batch_size, sequence_length)) and return a batched normalized probability distribution
over the vocabulary (i.e. torch.Tensor of shape (batch_size, sequence_length, vocab_size))
where the predicted distribution is over the next word for each input token.

STRUCTURE:
Inputs: (batch_size, sequence_length) of integer token IDs
v
(Token Embedding) -- convert token ids to dense vectors
v
(batch_size, sequence_length, d_model) of dense embedded tokens
v
(Transformer Block) x n -- apply attention
v
(batch_size, sequence_length, d_model) of dense embedded tokens
v
(LayerNorm) -- normalize
v
(batch_size, sequence_length, d_model) of dense embedded tokens
v
(Output Layer) -- turn into probability distribution
v
(batch_size, sequence_length, vocab_size) of embedded tokens
v
(Softmax) -- normalize probability distribution
v
(batch_size, sequence_length, vocab_size) of next-token probabilities
"""

from .linear import Linear
from .embedding import Embedding
from .normalization import RMSNorm
from .feedforward import SwiGLU, silu

__all__ = ["Linear", "Embedding", "RMSNorm", "SwiGLU", "silu"]
