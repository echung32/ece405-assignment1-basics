import torch
import torch.nn as nn
from jaxtyping import Float, Int

from cs336_basics.module.embedding import Embedding
from cs336_basics.module.transformer_block import TransformerBlock
from cs336_basics.module.rmsnorm import RMSNorm
from cs336_basics.module.linear import Linear


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model "Attention is All You Need" (Vaswani et al., 2017)

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        context_length: Maximum sequence length the model can process
        d_model: Dimensionality of embeddings and hidden states
        num_layers: Number of Transformer blocks to stack
        num_heads: Number of attention heads in each block
        d_ff: Dimensionality of the FFN inner layer in each block
        theta: RoPE base frequency parameter (default: 10000.0)
        device (torch.device | None): Device to store the parameters on
        dtype (torch.dtype | None): Data type of the parameters
    """

    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            theta: float = 10000.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        # token embeddings
        # in: (batch_size, sequence_length), out: (batch_size, sequence_length, d_model)
        self.token_embeddings = Embedding(num_embeddings=vocab_size,
                                          embedding_dim=d_model)

        # transformer blocks x num_layers
        # in: (batch, seq_len, d_model), out: (batch, seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
            )
            for _ in range(num_layers)
        ])

        # apply norm
        self.ln_final = RMSNorm(d_model=d_model)

        # linear output embedding
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(
            self,
            input_ids: Int[torch.Tensor, "batch seq_len"],
    ) -> Float[torch.Tensor, "batch seq_len vocab_size"]:
        """
        Forward pass through the Transformer language model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
        """

        # embed the input tokens
        x: Float[torch.Tensor, "batch seq_len d_model"] = self.token_embeddings(input_ids)

        # pass through all transformer blocks
        for layer in self.layers:
            x = layer(x)

        # apply rms norm
        x = self.ln_final(x)

        # linear output
        x = self.lm_head(x)

        return x

