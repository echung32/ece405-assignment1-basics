"""
Text generation (decoding) for language models.

Implements temperature-scaled softmax and nucleus (top-p) sampling for 
autoregressive text generation from Transformer language models.
"""
import torch
import torch.nn as nn
from jaxtyping import Int, Float
from typing import Optional

from cs336_basics.softmax import softmax


def softmax_with_temperature(
    logits: Float[torch.Tensor, "vocab_size"],
    temperature: float = 1.0
) -> Float[torch.Tensor, "vocab_size"]:
    """
    Apply softmax with temperature scaling.

    Args:
        logits: Unnormalized log-probabilities
        temperature: Temperature τ > 0. Lower = more peaked, higher = more uniform.
    """
    # softmax(v, τ)_i = exp(v_i/τ) / Σ_j exp(v_j/τ)
    # the inputs to softmax are divided by temperature
    scaled_logits = logits / temperature
    return softmax(scaled_logits, dim=-1)


def top_p_sampling(
    probs: Float[torch.Tensor, "vocab_size"],
    p: float = 1.0
) -> Float[torch.Tensor, "vocab_size"]:
    """
    Apply nucleus (top-p) sampling.
    
    Truncates distribution to smallest set V(p) where Σ_{j∈V(p)} q_j ≥ p.
    
    Args:
        probs: Probability distribution
        p: Cumulative probability threshold in (0, 1]
    """
    # truncating low probability words.
    # q is prob distribution from temperature-scaled softmax vocab_size
    # V(p) where Σ_{j∈V(p)} q_j ≥ p.
    # https://cyrilzakka.github.io/llm-playbook/nested/topp.html

    # first sort and compute cumulative probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # smallest set where cumsum >= p
    cutoff_index = torch.searchsorted(cumulative_probs, p, right=False)
    cutoff_index = min(cutoff_index.item() + 1, len(probs))
    
    # create mask for valid logits
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[sorted_indices[:cutoff_index]] = True
    
    # filter and renormalize
    filtered_probs = probs * mask.float()
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    return filtered_probs


def generate(
    model: nn.Module,
    prompt_tokens: Int[torch.Tensor, "seq_len"],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> Int[torch.Tensor, "generated_seq_len"]:
    """
    Generate text autoregressively from a language model.
    
    Args:
        model: Language model
        prompt_tokens: Initial sequence
        max_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (default: 1.0)
        top_p: Nucleus sampling threshold (default: 1.0 = no filtering)
        eos_token_id: Stop generation if this token is sampled
    """

    model.eval()
    device = next(model.parameters()).device
    
    current_sequence = prompt_tokens.to(device)
    
    # get context length from model
    context_length = model.context_length if hasattr(model, 'context_length') else None

    # P(x_{t+1} = i | x_{1...t}) = softmax(TransformerLM(x_{1...t})_t / τ)_i
    with torch.no_grad():
        for _ in range(max_tokens):
            # truncate sequence to context length
            if context_length is not None and current_sequence.size(0) > context_length:
                input_sequence = current_sequence[-context_length:]
            else:
                input_sequence = current_sequence
            
            # Forward pass: (1, seq_len) -> (1, seq_len, vocab_size)
            input_ids = input_sequence.unsqueeze(0)
            logits = model(input_ids)
            
            # Extract next-token logits: (vocab_size,)
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature scaling
            probs = softmax_with_temperature(next_token_logits, temperature=temperature)
            
            # Apply nucleus sampling
            filtered_probs = top_p_sampling(probs, p=top_p)
            
            # Sample next token
            next_token = torch.multinomial(filtered_probs, num_samples=1)
            
            # Append to sequence
            current_sequence = torch.cat([current_sequence, next_token], dim=-1)
            
            # Check for end-of-sequence
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    
    return current_sequence

