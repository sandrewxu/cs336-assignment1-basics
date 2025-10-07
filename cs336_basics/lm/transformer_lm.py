import torch
from cs336_basics.lm.attention import MultiHeadSelfAttention, softmax
from cs336_basics.lm.ff import positionwise_feedforward
from cs336_basics.lm.rmsnorm import RMSNorm
from cs336_basics.lm.embedding import Embedding
from cs336_basics.lm.linear import Linear

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = None, theta: float = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        use_rope = (theta and max_seq_len)
        self.attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, use_rope=use_rope)
        self.norm2 = RMSNorm(d_model)
        self.ffn = positionwise_feedforward(d_model=d_model, d_ff=d_ff)
    
    def forward(self, x: torch.Tensor):
        out = x + self.attention(self.norm1(x))
        out = out + self.ffn(self.norm2(out))
        return out

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float = None, softmax: bool = True):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.linear = Linear(in_features=d_model, out_features=vocab_size)
        self.softmax = softmax

    def forward(self, x: torch.Tensor):
        """
        Run an input tensor with shape (batch_size, seq_len) of token ids
        Get output probabilities (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = self.final_norm(x)
        x = self.linear(x)
        if self.softmax:
            x = softmax(x, -1) # softmax along dim=-1 (vocab_size)
        return x
