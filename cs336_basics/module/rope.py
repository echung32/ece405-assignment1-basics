import torch
from jaxtyping import Int, Float
from torch import nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) from Su et al., 2021.
    
    RoPE applies pairwise rotations to embedding elements based on token positions.
    For a query/key at position i with dimension d_k, we rotate pairs of elements
    by angles determined by the position and dimension index.
    
    The rotation angle for position i and dimension pair k is:
        θ(i,k) = i / Θ^((2k-2)/d)
    
    where Θ is a hyperparameter and k ∈ {1, ..., d_k/2}.
    
    This creates a block-diagonal rotation matrix with 2×2 rotation blocks:
        R_k^i = [[cos(θ), -sin(θ)],
                 [sin(θ),  cos(θ)]]
    
    Reference: RoFormer: Enhanced Transformer with Rotary Position Embedding
               (Su et al., 2021) - https://arxiv.org/abs/2104.09864
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and precompute sin/cos buffers.
        
        Args:
            theta (float): Θ value for RoPE
            d_k (int): Dimension of query and key vectors (must be even)
            max_seq_len (int): Maximum sequence length that will be processed
            device (torch.device | None): Device to store buffers on
        """
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for pairwise rotations"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # compute the frequency for each dimension pair k ∈ {1, ..., d_k/2}
        # The formula is: freq_k = 1 / (Θ^((2k-2)/d_k))
        # For k=1: freq = 1 / Θ^0 = 1
        # For k=2: freq = 1 / Θ^(2/d_k)
        # ...
        # "with blocks Ri
        # k for k ∈ {1, . . . , d/2}, with"

        k = torch.arange(1, d_k // 2 + 1)
        # print(k, k.shape)
        exponent = (2 * (k - 1)) / d_k
        # print(exponent, exponent.shape)
        freqs = 1.0 / (theta ** exponent)
        # print(freqs, freqs.shape)

        # create position indices for all positions [0, 1, ..., max_seq_len-1]
        positions: Int["max_seq_len"] = torch.arange(max_seq_len, device=device)
        # print(positions, positions.shape)

        # compute angles θ(i,k) = position * freq for all positions and freqs
        # Use outer product: angles[i, k] = positions[i] * freqs[k]
        angles: Float["max_seq_len d_k/2"] = torch.outer(positions, freqs)
        # print(angles, angles.shape)

        # precompute sin and cos values
        cos_cached: Float["max_seq_len d_k/2"] = torch.cos(angles)
        sin_cached: Float["max_seq_len d_k/2"] = torch.sin(angles)
        # print(cos_cached, cos_cached.shape, sin_cached, sin_cached.shape)

        # "and it can have a 2d pre-computed buffer of sin and cos values
        # created during init with self.register_buffer(persistent=False)"
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Float[torch.Tensor, "... seq_len"]) -> \
            Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply RoPE to the input tensor.
        
        The rotation is applied pairwise to consecutive dimensions:
        - Elements at indices (2k-1, 2k) are rotated together
        - x[..., 0:2] rotated by θ(i, 1)
        - x[..., 2:4] rotated by θ(i, 2)
        - etc.
        
        Efficient implementation WITHOUT constructing the full rotation matrix:
        For each pair [x1, x2] rotated by angle θ:
            x1' = x1 * cos(θ) - x2 * sin(θ)
            x2' = x1 * sin(θ) + x2 * cos(θ)
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k)
                Can have arbitrary batch dimensions
            token_positions (torch.Tensor): Token positions of shape (..., seq_len)
                Specifies the position index for each token
        
        Returns:
            torch.Tensor: Rotated tensor of the same shape (..., seq_len, d_k)
        """

        if token_positions is None:
            # generate positions from x's seq_len
            token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)

        # extract cos and sin values for the given token positions
        cos: Float["..., seq_len, d_k/2"] = self.cos_cached[token_positions]
        sin: Float["..., seq_len, d_k/2"] = self.sin_cached[token_positions]
        # print(cos.shape, sin.shape)

        # split x into pairs (even and odd indices)
        # r_i is a block-diagonal with 2x2 blocks of Ri_k (illustrated as 1 value but every value is a 2x2 block)
        # every block ri_k only acts on the subvector, so you have to split input vector x into pairs (even/odd)
        # then each pair is rotated independently by ri_k.
        x1 = x[..., 0::2]  # even
        x2 = x[..., 1::2]  # odd

        # apply rotation to each pair
        # The 2D rotation formula:
        # [x1']   [cos(θ)  -sin(θ)] [x1]
        # [x2'] = [sin(θ)   cos(θ)] [x2]

        x1_rotated: Float["..., seq_len, d_k/2"] = x1 * cos - x2 * sin
        x2_rotated: Float["..., seq_len, d_k/2"] = x1 * sin + x2 * cos

        # interleave the rotated pairs back together
        output: Float["..., seq_len, d_k"] = torch.empty_like(x)
        output[..., 0::2] = x1_rotated
        output[..., 1::2] = x2_rotated

        return output
