"""
Cross-entropy loss
"""

import math
import torch
from jaxtyping import Float, Int
from typing import Iterable

def cross_entropy(
    predicted_logits: Float[torch.Tensor, "... vocab_size"],
    targets: Int[torch.Tensor, "..."],
) -> Float[torch.Tensor, ""]:
    """
    Cross-entropy
    """
    norm_logits = predicted_logits - predicted_logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.logsumexp(norm_logits, dim=-1) # shape = [...]
    target_logits = norm_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) # shape [...]
    cross_entropies = logsumexp - target_logits
    return cross_entropies.mean()

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Gradient clipping
    """
    grad_norm_sq = 0
    for p in parameters:
        if p.grad is not None:
            grad_norm_sq += torch.sum(torch.square(p.grad.data))
    grad_norm = math.sqrt(grad_norm_sq)
    if grad_norm > max_l2_norm:
        scale = max_l2_norm / (grad_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= scale
