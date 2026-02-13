import einx
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    Compute scaled dot-product attention.
    
    The attention operation is defined as:
        Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
    
    Where:
        - Q is the query matrix (n × d_k)
        - K is the key matrix (m × d_k)
        - V is the value matrix (m × d_v)
        - n is the number of queries (sequence length for queries)
        - m is the number of keys/values (sequence length for keys/values)
        - d_k is the dimension of keys and queries
        - d_v is the dimension of values

    Args:
        Q: Query tensor of shape (..., n, d_k)
           Can have arbitrary batch dimensions before the last two dims
        K: Key tensor of shape (..., m, d_k)
           Must have same batch dims as Q
        V: Value tensor of shape (..., m, d_v)
           Must have same batch dims and sequence length (m) as K
        mask: Optional boolean mask of shape (..., n, m)
              - True = allow attention (information flows)
              - False = block attention (no information flows)
              If None, all positions can attend to all other positions
    
    Returns:
        Attention output of shape (..., n, d_v)
        Same batch dimensions as input, with n queries and d_v features
    """

    d_k = torch.tensor(Q.shape[-1], dtype=Q.dtype, device=Q.device)  # last dimension of Q

    # The attention scores tell us how much each query should attend to each key.
    # 
    # Mathematical intuition:
    #   - Q @ K^T computes similarity between each query-key pair
    #   - Shape: (..., n, d_k) @ (..., d_k, m) → (..., n, m)
    #   - Each element [i,j] is the dot product of query i with key j
    #   - Larger dot product = more similar = higher attention weight

    # with einx we don't have to transpose the K tensor first.
    scores: Float[Tensor, "... n m"] = einx.dot("... n [d_k], ... m [d_k] -> ... n m", Q, K) / torch.sqrt(d_k)

    # Masking prevents certain queries from attending to certain keys.
    # "Computationally, it will be much more efficient to use masking than to compute attention on subsequences,
    # and we can do this by taking the pre-softmax values and adding a −∞ in any entry of the mask matrix that is False."
    # after softmax the exp(-inf) = 0 so the positions get 0 weight. if it was exp(0) then it would still get weight.
    if mask is not None:
        # when the value is True, then it does attend to the key.
        scores = scores.masked_fill(~mask, float('-inf'))

    # softmax over the keys dimension (m)
    attention_weights: Float[Tensor, "... n m"] = softmax(scores, dim=-1)

    # attention_weights is ... n m and V is (..., m, d_v). "output with the shape (batch_size, ..., d_v)"
    output: Float[Tensor, "... n d_v"] = einx.dot("... n [m], ... [m] d_v -> ... n d_v", attention_weights, V)

    return output
