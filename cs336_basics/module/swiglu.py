import torch
from torch import nn
from jaxtyping import Float

from cs336_basics.module.linear import Linear


class SiLU(nn.Module):
    """
    The SiLU or Swish activation function [Hendrycks and Gimpel, 2016, Elfwing et al., 2017]
    the SiLU activation function is similar to the ReLU activation function, but is smooth at zero
    """

    def forward(self, x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
        """
        Apply SiLU activation.
        
        Args:
            x (torch.Tensor): Input tensor of arbitrary shape
        """
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network combining SiLU activation with Gated Linear Units (GLU).
    
    FFN(x) = W2(SiLU(W1·x) ⊙ W3·x)
    
    where:
    - x ∈ ℝ^(d_model)
    - W1, W3 ∈ ℝ^(d_ff × d_model)
    - W2 ∈ ℝ^(d_model × d_ff)
    - ⊙ represents element-wise multiplication
    - Canonically, d_ff = (8/3) * d_model, rounded to nearest multiple of 64
    
    Reference: Shazeer, 2020; used in LLaMA (Touvron et al., 2023), 
               Llama 3 (Grattafiori et al., 2024), Qwen 2.5 (Yang et al., 2024)
    """

    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        """
        Construct the SwiGLU feed-forward network.
        
        Args:
            d_model (int): Dimensionality of the input and output
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model

        if d_ff is None:
            # set d_ff to approximately (8/3) * d_model
            d_ff = int(8 * d_model / 3)

        # round to the nearest multiple of 64
        d_ff = ((d_ff + 64 // 2) // 64) * 64

        self.d_ff = d_ff

        # W1: d_model -> d_ff (for the SiLU branch)
        # W2: d_ff -> d_model (output projection)
        # W3: d_model -> d_ff (for the linear branch in the gate)

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

        self.silu = SiLU()

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        """
        Apply the SwiGLU feed-forward network.
        
        The computation follows:
        1. Compute SiLU(W1·x)
        2. Compute W3·x
        3. Element-wise multiply: SiLU(W1·x) ⊙ W3·x
        4. Apply output projection: W2(...)
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)
        """

        gate = self.silu(self.w1(x))
        linear = self.w3(x)
        gated = gate * linear
        output = self.w2(gated)
        return output
