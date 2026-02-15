from typing import Iterable
import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """

    # collect all params with gradients
    params_with_grad = [p for p in parameters if p.grad is not None]

    if len(params_with_grad) == 0:
        return
    
    # compute the total L2 norm of all gradients ||g||₂
    total_norm = torch.sqrt(
        sum(p.grad.norm(2).pow(2) for p in params_with_grad)
    )

    # If this norm is less than a
    # maximum value M, then we leave g as is; otherwise, we scale g down by a factor of M
    # ∥g∥2+ϵ (where a small ϵ, like 10−6, is added for numeric stability)
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in params_with_grad:
            p.grad.mul_(clip_coef)
