"""
Ablation 1b: Post-norm Transformer LM.
"""
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from cs336_basics.module.embedding import Embedding
from cs336_basics.module.linear import Linear
from cs336_basics.ablation.post_norm_transformer_block import PostNormTransformerBlock


class PostNormTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, theta: float = 10000.0):
        super().__init__()

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.layers = nn.ModuleList([
            PostNormTransformerBlock(
                d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                max_seq_len=context_length, theta=theta,
            )
            for _ in range(num_layers)
        ])

        # skip the final RMSNorm needed for post-norm cus already normalized in each block

        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, input_ids: Int[torch.Tensor, "batch seq_len"]) -> Float[
        torch.Tensor, "batch seq_len vocab_size"]:
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.lm_head(x)
        return x
