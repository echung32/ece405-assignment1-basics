import math


def get_lr_cosine_schedule(
    iteration: int, # t
    max_learning_rate: float, # alpha_max
    min_learning_rate: float, # alpha_min
    warmup_iters: int, # t_w
    cosine_cycle_iters: int, # t_c
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.
    
    Args:
        iteration (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # warmup if t < T_w, α_t = (t / T_w) × α_max, linear increase from 0 to t_w
    if iteration < warmup_iters:
        return (iteration / warmup_iters) * max_learning_rate
    
    # cosine annealing if T_w ≤ t ≤ T_c, smooth decay
    #  α_t = α_min + 0.5 × (1 + cos((t - T_w) / (T_c - T_w) × π)) × (α_max - α_min)
    elif iteration <= cosine_cycle_iters:
        # (t - T_w) / (T_c - T_w)
        # t = T_w: progress = 0
        # t = T_c: progress = 1
        progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)

        # α_t = α_min + 0.5 × (1 + cos(progress × π)) × (α_max - α_min)
        # progress = 0: cos(0) = 1, so α_t = α_max
        # progress = 1: cos(π) = -1, so α_t = α_min
        cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    
    # post-annealing t > T_c then α_t = α_min, constant minimum lr
    else:
        return min_learning_rate

