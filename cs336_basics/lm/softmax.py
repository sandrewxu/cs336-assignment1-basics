import torch

def softmax(x: torch.Tensor, i: int):
    """
    Apply softmax to the i-th dimension of input tensor x
    """
    x_max = torch.max(x, dim=i, keepdim=True).values
    x_mod = x - x_max
    exp_x = torch.exp(x_mod)
    x_sum = torch.sum(exp_x, dim=i, keepdim=True)
    result = exp_x / x_sum
    return result
