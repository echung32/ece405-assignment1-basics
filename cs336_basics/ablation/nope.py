"""
Ablation 2: mhsa with no position embeddings.

It turns out that decoder-only transformers, i.e., those with a causal
mask as we have implemented, can in theory infer relative or absolute position information without being
provided with position embeddings explicitly [Tsai et al., 2019, Kazemnejad et al., 2023]. We will now test
empirically how NoPE performs compared to RoPE.
"""
import einx
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from cs336_basics.module.linear import Linear
from cs336_basics.attention import scaled_dot_product_attention


class NoPEMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(in_features=d_model, out_features=num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=num_heads * self.d_k, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=num_heads * self.d_k, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        batch_size, seq_len, _ = x.shape

        Q = einx.rearrange("b s (h d) -> b h s d", self.q_proj(x), h=self.num_heads)
        K = einx.rearrange("b s (h d) -> b h s d", self.k_proj(x), h=self.num_heads)
        V = einx.rearrange("b s (h d) -> b h s d", self.v_proj(x), h=self.num_heads)

        # No RoPE applied

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device))
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = einx.rearrange("b h s d -> b s (h d)", attn_output)
        return self.output_proj(attn_output)
