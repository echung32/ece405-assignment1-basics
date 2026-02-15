from collections.abc import Callable
from typing import Optional

import numpy as np
import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    Implements the AdamW algorithm as described in Loshchilov and Hutter [2019].
    Algorithm from Section 4.3 of the assignment.
    
    The algorithm maintains first and second moment estimates (m and v) and performs
    the following updates at each step:
        1. m ← β₁m + (1 - β₁)g
        2. v ← β₂v + (1 - β₂)g²
        3. α_t ← α * √(1 - β₂^t) / (1 - β₁^t)
        4. θ ← θ - α_t * m / (√v + ε)
        5. θ ← θ - αλθ (weight decay)
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate α (default: 1e-3)
        betas: Coefficients (β₁, β₂) for computing running averages of gradient 
               and its square (default: (0.9, 0.999))
        eps: Term ε added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient λ (default: 0.01)
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            # modern lms like llama or gpt-3 usually trained with 0.9/0.95
            # typical applications use range 0.9 to 0.999.
            betas: tuple[float, float] = (0.9, 0.999),
            # here ϵ is a small value (e.g., 10−8) used to improve numerical stability
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    # implement step overloaed from optimizer
    # not sure if we need the closure in assignment but linter gives warning
    def step(self, closure: Optional[Callable] = None) -> None:
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            The loss value if closure is provided, otherwise None
        """
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # get state for param, otherwise init if not exist
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0  # this is the time step.
                    # "Initial value of the first moment vector; same shape as θ"
                    state["m"] = torch.zeros_like(p.data)
                    # "Initial value of the second moment vector; same shape as θ"
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                grad = p.grad.data

                # increment time step (1...T)
                state["t"] += 1
                t = state["t"]

                # update the first moment estimate: m ← β₁m + (1 - β₁)g
                # m = beta1 * m + (1 - beta1) * grad
                # need to update them in-place, above won't work
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # update second moment estimate: v ← β₂v + (1 - β₂)g²
                # v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute adjusted α for iteration t α_t ← α * √(1 - β₂^t) / (1 - β₁^t)
                lr_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # update parameters θ ← θ - α_t * m / (√v + ε)
                # p = p - lr_t * m / (torch.sqrt(v) + eps)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

                # weight decay θ ← θ - αλθ
                # p = p - lr * weight_decay * grad
                p.data.mul_(1 - lr * weight_decay)
