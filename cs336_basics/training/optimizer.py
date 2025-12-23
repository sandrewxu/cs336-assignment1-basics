"""
Optimizers: SGD, AdamW
"""

from collections.abc import Callable, Iterable
import math
import torch
import torch.optim as optim
from typing import Optional

class SGD(optim.Optimizer):
    """
    Stochastic gradient descent optimizer.
    """

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(optim.Optimizer):
    """
    AdamW optimizer
    """

    def __init__(self, params, lr=1e-3, betas=(0.90, 0.95), eps=1e-8, weight_decay=1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                grad = p.grad.data

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2
                lr_t = lr * math.sqrt(1-math.pow(beta_2, t)) / (1 - math.pow(beta_1, t))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["m"], state["v"] = m, v
        return loss

if __name__ == "__main__":
    # 4.2 Learning Rate Tuning for SGD
    for lr in 10, 100, 1000:
        print(f"=== LR={lr} ===")
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)

        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(loss.cpu().item())
            loss.backward()
            opt.step()
