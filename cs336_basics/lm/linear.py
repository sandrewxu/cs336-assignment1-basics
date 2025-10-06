import torch
import math
from einops import einsum

class Linear(torch.nn.Module):
    """
    Linear class that performs a linear transformation.
    Follows the interface of PyTorch's built-in nn.Linear module,
    except for not having a bias argument or parameter.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
