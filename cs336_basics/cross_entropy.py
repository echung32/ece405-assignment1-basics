import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    predicted: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """
    Compute the average cross-entropy loss across examples.
    
    Given logits o ∈ ℝ^vocab_size and target token index x, cross-entropy is:
        ℓ = -log softmax(o)[x] = -log(exp(o[x]) / Σ_a exp(o[a]))
    
    This simplifies to:
        ℓ = -o[x] + log(Σ_a exp(o[a]))
    
    For numerical stability, we subtract the max:
        ℓ = -o[x] + log(Σ_a exp(o[a] - max(o))) + max(o)

    Args:
        predicted: Unnormalized logits of shape (batch_size, vocab_size)
                inputs[i][j] is the logit for class j in example i
        targets: Target class indices of shape (batch_size,)
                Each value must be in [0, vocab_size - 1]
    """

    # get the max of the predicted logits
    max_logits: Float[torch.Tensor, "batch_size 1"] = torch.max(predicted, dim=-1, keepdim=True).values
    # print(max_logits.shape)

    # subtract largest element for numerical stability
    shifted_logits: Float[torch.Tensor, "batch_size vocab_size"] = predicted - max_logits
    # print(shifted_logits.shape)
    
    # compute log probability
    log_sum_exp: Float[torch.Tensor, "batch_size 1"] = torch.logsumexp(shifted_logits, dim=-1, keepdim=True)
    # print(log_sum_exp.shape)
    
    # gather target log probabilities by their indices
    target_logits: Float[torch.Tensor, "batch_size 1"] = torch.gather(predicted, dim=-1, index=targets.unsqueeze(-1))
    # print(target_logits.shape)

    # cross-entropy per example
    loss_per_example: Float[torch.Tensor, "batch_size 1"] = -target_logits + max_logits + log_sum_exp
    # print(loss_per_example.shape)

    # calculate the mean
    return loss_per_example.squeeze(-1).mean()


def perplexity(cross_entrypy_loss: Float[Tensor, "..."]) -> Float[Tensor, ""]:
    """
    Cross entropy suffices for training, but when we evaluate the model, we also want to report
    perplexity. For a sequence of length m where we suffer cross-entropy losses ℓ1, . . . , ℓm:
    perplexity = exp(1/m sum {m, i=1} ℓi

    essentially perplexity is just exp of cross entropy
    """
    return torch.exp(cross_entrypy_loss)