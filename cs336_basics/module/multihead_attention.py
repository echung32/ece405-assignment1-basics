import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.module.linear import Linear
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.module.rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal masking.
    
    This module implements the multi-head attention mechanism from
    "Attention is All You Need" (Vaswani et al., 2017).
    
    Mathematical Definition:
    ========================
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
    
    where head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
    
    For self-attention: Q = K = V = x (the input)
    
    So: MultiHeadSelfAttention(x) = W_O @ MultiHead(W_Q @ x, W_K @ x, W_V @ x)
    
    Parameters:
    ===========
    - W_Q ∈ R^(h*d_k × d_model): Query projection (for all heads combined)
    - W_K ∈ R^(h*d_k × d_model): Key projection (for all heads combined)
    - W_V ∈ R^(h*d_v × d_model): Value projection (for all heads combined)
    - W_O ∈ R^(d_model × h*d_v): Output projection
    
    where:
    - h = num_heads
    - d_k = d_v = d_model / h (dimension per head)

    Args:
        d_model: Dimensionality of input/output embeddings
        num_heads: Number of attention heads
        use_rope: Whether to apply Rotary Position Embeddings (RoPE)
                  Set to False for basic multi-head attention
        max_seq_len: Maximum sequence length (needed for RoPE)
        theta: RoPE base frequency (needed for RoPE)
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            use_rope: bool = False,
            max_seq_len: int | None = None,
            theta: float = 10000.0,
    ):
        super().__init__()

        # make sure d_model can be split evenly across all heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope

        # "Following Vaswani et al. [2017], set dk = dv = dmodel/h."
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Here, the learnable parameters are WQ ∈ Rhdk ×dmodel , WK ∈ Rhdk ×dmodel , WV ∈ Rhdv ×dmodel
        self.q_proj = Linear(in_features=d_model, out_features=num_heads * self.d_k)
        self.k_proj = Linear(in_features=d_model, out_features=num_heads * self.d_k)
        self.v_proj = Linear(in_features=d_model, out_features=num_heads * self.d_v)

        # WO ∈ Rdmodel×hdv .
        self.output_proj = Linear(in_features=num_heads * self.d_v, out_features=d_model)

        # RoPE should be applied to the query and key vectors, but not the value vectors.
        if use_rope:
            assert max_seq_len is not None, "max_seq_len not provided"
            self.rope = RotaryPositionalEmbedding(
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                theta=theta,
            )
        else:
            self.rope = None

    def _create_causal_mask(
            self,
            seq_len: int,
            device: torch.device,
    ) -> Bool[Tensor, "seq_len seq_len"]:
        """
        Create a causal attention mask.
        
        The mask is lower-triangular: position 'i' can attend to positions j ≤ i.
        
        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on
        
        Returns:
            Boolean mask of shape (seq_len, seq_len) where:
            - True means "can attend" (information flows)
            - False means "cannot attend" (blocked)
        """

        # we’ll use causal attention masking, which allows token i to attend to all
        # positions j ≤ i in the sequence. You can use torch.triu.

        # the assignment says to use triu but is that wrong? because then it would be upper triangular,
        # this would give true in lower triangle, in attention it is inverted and fills where mask is false.
        # so then the false positions get blocked, which i think is right (also passes the tests).
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    def forward(
            self,
            x: Float[Tensor, "batch seq_len d_model"],
            token_positions: Int[Tensor, "batch seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        Apply multi-head self-attention to input x.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices for RoPE
                            Shape: (batch, seq_len)
                            Not needed for basic attention (use_rope=False)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """

        batch_size, seq_len, _ = x.shape

        # 1. Project x to Q, K, V
        Q: Float[Tensor, "batch seq_len d_model"] = self.q_proj(x)
        K: Float[Tensor, "batch seq_len d_model"] = self.k_proj(x)
        V: Float[Tensor, "batch seq_len d_model"] = self.v_proj(x)

        # 2. Reshape to separate heads
        # to separate the heads, take out h from d_model
        # because they were made from WQ ∈ Rhdk ×dmodel , WK ∈ Rhdk ×dmodel , WV ∈ Rhdv ×dmodel
        # head has to be 2nd bc scaled_dot_product_attention expects last 2 to be nm d for qkv
        Q = einx.rearrange("b s (h d) -> b h s d", Q, h=self.num_heads)
        K = einx.rearrange("b s (h d) -> b h s d", K, h=self.num_heads)
        V = einx.rearrange("b s (h d) -> b h s d", V, h=self.num_heads)

        # 3. Apply RoPE
        if self.use_rope:
            # rope only applied on d_k, so q and k
            # this should be returning tensor of same shape
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 4. create casual mask
        mask: Float[Tensor, "seq_len seq_len"] = self._create_causal_mask(seq_len, device=Q.device)

        # 5. Apply scaled dot-product attention
        attn_output: Float[Tensor, "batch heads seq_len q_kv"] = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 6. Concatenate heads
        attn_output = einx.rearrange("b h s d -> b s (h d)", attn_output)

        # Last. Apply output projection
        output: Float[Tensor, "batch seq_len d_model"] = self.output_proj(attn_output)

        return output
