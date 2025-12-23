"""
Learning rate scheduling
"""

import math

def learning_rate_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """
    Returns learning rate given input parameters

    Args:
        t: int current iteration
        lr_max: float maximum learning rate
        lr_min: float minimum learning rate
        T_w: int warmup steps
        T_c: int cooldown threshold 
    """
    assert T_w <= T_c
    assert lr_min <= lr_max
    if t < T_w:
        return t/T_w * lr_max
    elif t > T_c:
        return lr_min
    else:
        return lr_min + 1/2 * (1 + math.cos((t - T_w)/(T_c-T_w) * math.pi)) * (lr_max - lr_min)
