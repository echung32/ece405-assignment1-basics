"""
Ablation 3: SiLU FFN (no gated linear unit GLU)

    FFN(x) = W2(SiLU(W1 @ x))

Recall that in our SwiGLU implementation, we set the dimensionality of the inner feed-forward layer to
be roughly dff = 8/3d_model (while ensuring that dff mod 64 = 0, to make use of GPU tensor cores). In your
FFNSiLU implementation you should set dff = 4 × dmodel, to approximately match the parameter count of
the SwiGLU feed-forward network (which has three instead of two weight matrices).
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.module.linear import Linear
from cs336_basics.module.swiglu import SiLU


class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()

        # hard-code d_ff to 4*d_model, ignore d_ff parameter.
        d_ff = 4 * d_model

        # round to the nearest multiple of 64
        d_ff = ((d_ff + 64 // 2) // 64) * 64

        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        return self.w2(self.silu(self.w1(x)))
