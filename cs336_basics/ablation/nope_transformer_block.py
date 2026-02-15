"""
Ablation 2: Transformer block using NoPE (no position embeddings).
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.ablation.nope import NoPEMultiHeadSelfAttention
from cs336_basics.module.rmsnorm import RMSNorm
from cs336_basics.module.swiglu import SwiGLU


class NoPETransformerBlock(nn.Module):
    """Pre-norm transformer block with NO position embeddings."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = NoPEMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len d_model"]) -> Float[
        torch.Tensor, "batch_size seq_len d_model"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
