"""
Model checkpointing
"""

import os
import torch
from typing import BinaryIO, IO

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Dump all state from model, optimizer, and iteration into the file-like object out.
    Can use state_dict object of both the model and optimizer to get their relevant states.
    Use torch.save(obj, out) to dump obj into out
    """
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Load a checkpoint from src, recover the model and optimizer states from this checkpoint
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
