import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_states = model.state_dict()
    optimizer_states = optimizer.state_dict()
    obj = {"iteration": iteration, "model": model_states, "optimizer": optimizer_states}
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
