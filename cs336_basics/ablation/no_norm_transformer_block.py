"""
Ablation 1a: Transformer block with no RMSNorm.

It is often said that layer normalization is important for the stability
of Transformer training. But perhaps we want to live dangerously. Let’s remove RMSNorm from each of
our Transformer blocks and see what happens.
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.module.multihead_attention import MultiHeadSelfAttention
from cs336_basics.module.swiglu import SwiGLU


class NoNormTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None,
                 dtype=None):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads, use_rope=True, max_seq_len=max_seq_len, theta=theta,
                                           device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len d_model"]) -> Float[
        torch.Tensor, "batch_size seq_len d_model"]:
        # No normalization
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x
