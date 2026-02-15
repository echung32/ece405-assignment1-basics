"""
Ablation 3: Transformer block using SiLU FFN
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.module.multihead_attention import MultiHeadSelfAttention
from cs336_basics.module.rmsnorm import RMSNorm
from cs336_basics.ablation.silu_ffn import SiLUFFN


class SiLUTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None,
                 dtype=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, use_rope=True, max_seq_len=max_seq_len, theta=theta,
                                           device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SiLUFFN(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len d_model"]) -> Float[
        torch.Tensor, "batch_size seq_len d_model"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
