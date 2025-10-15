import math

def learning_rate_schedule(t, a_max, a_min, T_w, T_c):
    """
    returns learning rate a_t
    """
    if t < T_w:
        a_t = t/T_w * a_max
    elif t <= T_c:
        a_t = a_min + 0.5 * (1 + math.cos((t-T_w)/(T_c-T_w)*math.pi))*(a_max - a_min)
    else:
        a_t = a_min

    return a_t
