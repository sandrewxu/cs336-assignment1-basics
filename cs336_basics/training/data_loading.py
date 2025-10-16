import numpy as np
import torch

def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu"):
    """
    returns a pair of tensors: the sampled input sequences and the 
    corresponding next-token targets. Both tensors should have shape 
    (batch_size, context_length) containing token IDs, and both should be
    placed on the requested device.
    """
    # sample i uniformly from 1 to n-m B times, but uniquely
    n = len(x)
    m = context_length
    B = batch_size
    # i goes from 1 to n-m (inclusive), B unique samples
    starts = np.random.choice(np.arange(n - m), size=B, replace=False)

    # For each index in starts, get slice x[s:s+m] as input, x[s+1:s+m+1] as target
    samples = np.stack([x[s : s + m] for s in starts])
    targets = np.stack([x[s + 1 : s + m + 1] for s in starts])
    samples = torch.tensor(samples, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return samples, targets
