import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.G = torch.nn.Parameter(
            torch.ones(d_model)
        )

    def forward(self, x: torch.Tensor):
        """
        RMSNorm an input tensor of shape (batch_size, sequence_length, d_model).
        """
        # upcast input to torch32
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # calculate RMS, along each (1, d_model)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        
        # normalize
        result = x / rms

        # broadcast G
        result = result * self.G

        return result.to(in_dtype)