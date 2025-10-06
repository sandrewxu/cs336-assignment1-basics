import torch

class Embedding(torch.nn.Module):
    """
    Embedding class that performs an embedding lookup.
    Follows the interface of PyTorch's built-in nn.Embedding module.
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        index self.W by token_ids to get the vector representations
        """
        return self.W[token_ids]
