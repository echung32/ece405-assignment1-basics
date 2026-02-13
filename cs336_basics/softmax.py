import torch
from jaxtyping import Float


def softmax(input: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    """
    Apply the softmax operation on a tensor along a specified dimension.
    
    Softmax converts unnormalized logits into a probability distribution:
        softmax(v)_i = exp(v_i) / Σ_j exp(v_j)
    
    Args:
        input (torch.Tensor): Input tensor of any shape
        dim (int): Dimension along which to apply softmax
                   Can be negative (e.g., -1 for last dimension)

    Returns:
        torch.Tensor: The output tensor should have the same shape as the input tensor,
            but its i-th dimension will now have a normalized probability distribution.
    """
    
    # find max val
    max_val = torch.max(input, dim=dim, keepdim=True)[0]

    # note that exp(vi) can become inf for large values (then, inf/inf = NaN). We can avoid this by noticing
    # that the softmax operation is invariant to adding any constant c to all inputs. We can leverage this property
    # for numerical stability—typically, we will subtract the largest entry of oi from all elements of oi, making the
    # new largest entry 0.
    stable_input = input - max_val
    exp_input = torch.exp(stable_input)

    output = exp_input / torch.sum(exp_input, dim=dim, keepdim=True)

    return output

