import torch
from jaxtyping import Int, Float
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype=None) -> None:
        """
        Construct an embedding module.

        Args:
            num_embeddings (int): Size of the Vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e. d_model
            device: torch.device | None - Device to store the parameters on
            dtype: torch.dtype | None - Data type of the parameters
        """
        super().__init__()

        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # initialize weights using truncated normal distribution
        # σ² = 1
        std = 1
        # Truncate at [-3, 3]
        with torch.no_grad():
            nn.init.trunc_normal_(self.embedding, mean=0.0, std=std, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ... d_model"]:
        """
        Lookup the embedding vectors for the given token IDs

        the forward method should select the embedding
        vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
        torch.LongTensor of token IDs with shape (batch_size, sequence_length).

        Args:
            token_ids (torch.Tensor): Tensor of token IDs
        """
        return self.embedding[token_ids]

if __name__ == "__main__":
    vocab_size, d_model = 1000, 128
    emb = Embedding(vocab_size, d_model)
    assert emb.embedding.shape == (vocab_size, d_model)
    token_ids_1d = torch.randint(0, vocab_size, (10,))
    token_ids_2d = torch.randint(0, vocab_size, (4, 10))
    out_1d = emb(token_ids_1d)
    out_2d = emb(token_ids_2d)
    assert out_1d.shape == (10, d_model)
    assert out_2d.shape == (4, 10, d_model)