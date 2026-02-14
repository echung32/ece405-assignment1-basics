import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.module.multihead_attention import MultiHeadSelfAttention
from cs336_basics.module.rmsnorm import RMSNorm
from cs336_basics.module.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    pre-norm transformer block

    An intuition for pre-norm is that there is a clean “residual
    stream” without any normalization going from the input embeddings to the final output of the Transformer,
    which is purported to improve gradient flow.

    now standard in language models today (llama, gpt-3, palm).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None,
                 dtype=None):
        """
        Construct a pre-norm transformer block module.

        Args:
            d_model: int - Dimensionality of the Transformer block inputs
            num_heads: int - Number of heads to use in multi-head self-attention
            d_ff: int - Dimensionality of the position-wise feed-forward inner layer
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            device: torch.device | None - Device to store the parameters on
            dtype: torch.dtype | None - Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_iff = d_ff

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, use_rope=True, max_seq_len=max_seq_len, theta=theta,
                                           device=device, dtype=dtype)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len d_model"]) -> Float[
        torch.Tensor, "batch_size seq_len d_model"]:
        """
        Apply the pre-norm to the input.

        Args:
            x: torch.Tensor - input tensor with shape (batch_size seq_len d_model)

        Returns:
            torch.Tensor - output tensor with the same shape
        """

        # based on figure 2
        norm_ln1 = self.ln1(x)
        attn = self.attn(norm_ln1)
        x = x + attn

        norm_ln2 = self.ln2(x)
        ffn = self.ffn(norm_ln2)
        x = x + ffn

        return x
