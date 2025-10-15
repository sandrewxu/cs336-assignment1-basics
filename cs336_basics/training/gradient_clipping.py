from collections.abc import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    total_norm = 0
    for param in parameters:
        if param.grad is not None:
            weights = param.grad.data
            total_norm += torch.sum(torch.square(weights))
    
    total_norm = torch.sqrt(total_norm)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.data *= scale
