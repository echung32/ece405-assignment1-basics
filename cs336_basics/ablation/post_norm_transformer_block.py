"""
Ablation 1b: Post-norm Transformer block.

Uses post-norm instead of pre-norm:
    z = RMSNorm(x + MultiHeadedSelfAttention(x))
    y = RMSNorm(z + FFN(z))

This is the original Transformer architecture
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.module.multihead_attention import MultiHeadSelfAttention
from cs336_basics.module.rmsnorm import RMSNorm
from cs336_basics.module.swiglu import SwiGLU


class PostNormTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None,
                 dtype=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, use_rope=True, max_seq_len=max_seq_len, theta=theta,
                                           device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len d_model"]) -> Float[
        torch.Tensor, "batch_size seq_len d_model"]:
        # norm after residual addition
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x
