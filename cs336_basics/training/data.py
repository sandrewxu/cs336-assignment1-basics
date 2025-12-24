"""
Data loading
"""

import numpy as np
import torch
from typing import Optional

def data_loading(
    x: np.array,
    batch_size: int,
    context_length: int,
    device: str,
    seed: Optional[int] = None,
):
    """
    Takes a numpy array x (integer array with token IDs), a
    batch_size, a context_length, and a PyTorch device string
    (e.g., 'cpu' or 'cuda:0'), and returns a pair of tensors:
    the sampled input sequences and corresponding next-token targets.
    Both tensors should have shape (batch_size, context_length) containing
    token IDs, and both should be places on the requested device.
    """
    max_idx = x.size - context_length
    # sample from between 0 and max_idx
    rng = np.random.default_rng(seed=seed)
    starts = rng.choice(max_idx, size=batch_size, replace=False)
    
    # want to return two (B, C) arrays, where ins[B] = x[B:B+C], outs[B] = x[B+1:B+C+1]
    n_inputs = np.stack([x[s:s+context_length] for s in starts], axis=0)
    n_targets = np.stack([x[s+1:s+context_length+1] for s in starts], axis=0)
    inputs = torch.from_numpy(n_inputs).to(device)
    targets = torch.from_numpy(n_targets).to(device)
    return inputs, targets
