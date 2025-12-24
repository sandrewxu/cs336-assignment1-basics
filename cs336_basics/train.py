"""
Training Loop

Train a language model on user-provided input.
Includes the following features
- N/A

"""

import numpy as np
import os
from pathlib import Path
import torch
from typing import Callable, Optional
import wandb

from .model.transformer import TransformerLM
from .training.loss import cross_entropy
from .training.optimizer import AdamW
from .training.checkpointing import save_checkpoint
from .training.data import data_loading
from .training.scheduler import learning_rate_schedule

def main(
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    train_data_path: str = "data",
    val_data_path: str = "data",
    start_step: int = 0,
    end_step: int = 5000,
    batch_size: int = 256,
    context_length: int = 256,
    output_dir_base: str = "/gpfs/radev/home/ax46/scratch/A1",
    steps_per_eval: int = 10,
    steps_per_checkpoint: int = 1000,
    vocab_size: int = 10000,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    rope_theta: float = 10000,
    betas: tuple[float, float] = (0.90, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1,
    min_lr: float = 1e-3,
    max_lr: float = 1e-3,
    warmup_steps: float = 500,
    cooldown_steps: int = 1000,
    device: str = "cuda:0",
):
    """
    Given a path-like object for model (or initialize from scratch).
    Configure and control model and optimizer hyperparameters (or load at a certain state)
    Assume that data is saved using np.save()
    """
    # Load model, optimizer, loss function
    if model is not None:
        model = model
    else:
        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=device,
        )

    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = AdamW(
            params=model.parameters(),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            device=device,
        )

    if loss_fn is not None:
        loss_fn = loss_fn
    else:
        loss_fn = cross_entropy

    # Load train and validation data
    train_data = np.load(train_data_path, mmap_mode='r')
    val_data = np.load(val_data_path, mmap_mode='r')

    # Call train_lm
    train_lm(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        start_step=start_step,
        end_step=end_step,
        batch_size=batch_size,
        context_length=context_length,
        lr_min=min_lr,
        lr_max=max_lr,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        output_dir_base=output_dir_base,
        steps_per_eval=steps_per_eval,
        steps_per_checkpoint=steps_per_checkpoint,
        device=device,
    )

def train_lm(
    model: torch.nn.Module,
    optimizer: torch.nn.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_data: np.array,
    val_data: np.array,
    start_step: int,
    end_step: int,
    batch_size: int,
    context_length: int,
    lr_max: int,
    lr_min: int,
    warmup_steps: int,
    cooldown_steps: int,
    output_dir_base: str,
    steps_per_eval: int,
    steps_per_checkpoint: Optional[int],
    device: str,
):
    """
    Given an input dataset, batch_size, sequence_length, epochs, optimizer & model, train an LM.
    """
    output_dir_base = Path(output_dir_base)
    output_dir_base.mkdir(parents=True, exist_ok=True)
    for step in range(start_step, end_step):
        current_lr = learning_rate_schedule(
            t=step,
            lr_max=lr_max,
            lr_min=lr_min,
            T_w=warmup_steps,
            T_c=end_step-cooldown_steps,
        )
        optimizer.param_groups[0]["lr"] = current_lr
        inputs, targets = data_loading(train_data, batch_size, context_length, device)
        predicted_logits = model(inputs) # (batch_size, sequence_length)
        loss = loss_fn(predicted_logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if wandb.run is not None:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": current_lr,
                "train/step": step + 1,
            })

        # Occasionally log validation metrics (in WandB if available)
        if (step + 1) % steps_per_eval == 0 or step == end_step - 1:
            with torch.inference_mode():
                val_inputs, val_targets = data_loading(val_data, batch_size, context_length, device)
                val_logits = model(val_inputs)
                val_loss = loss_fn(val_logits, val_targets)

            if wandb.run is not None:
                wandb.log({
                    "eval/loss": val_loss.item(),
                    "eval/step": step + 1,
                })
            else:
                print(f"=== Step {step + 1} ===")
                print(f"Training loss: {loss.item()}")
                print(f"Validation loss: {val_loss.item()}")

        # Occasional checkpointing
        if (step + 1) % steps_per_checkpoint == 0 or step == end_step - 1:
            output_path = output_dir_base / f"step_{step + 1}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step + 1,
                out=output_path,
            )
