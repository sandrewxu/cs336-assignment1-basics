import torch
import math
from einops import einsum

class positionwise_feedforward(torch.nn.Module):
    """
    Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.
    FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) ⊙ W3x) 
    """
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        if not d_ff:
            d_ff = round((8/3) * d_model / 64) * 64
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model)
        )
        self.W2 = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_ff)
        )
        self.W3 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model)
        )

        std = math.sqrt(2/(d_ff+d_model))
        torch.nn.init.trunc_normal_(self.W1, mean=0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.W2, mean=0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.W3, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor):
        def SiLU(y: torch.Tensor):
            return y * torch.sigmoid(y)
        result = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        result = SiLU(result)
        result = result * einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        result = einsum(self.W2, result, "d_model d_ff, ... d_ff -> ... d_model")
        return result
