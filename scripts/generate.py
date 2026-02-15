"""
uv run python scripts/generate.py \
    --checkpoint checkpoints/checkpoint_latest.pt \
    --vocab-file artifacts/tinystories_vocab.json \
    --merges-file artifacts/tinystories_merges.txt \
    --vocab-size 10000 \
    --prompt "Once upon a time"
"""

import torch
import typer
from typing_extensions import Annotated
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.generation import generate


app = typer.Typer()


@app.command()
def main(
    checkpoint: Annotated[str, typer.Option(help="Path to model checkpoint")],
    vocab_file: Annotated[str, typer.Option(help="Path to vocab.json")],
    merges_file: Annotated[str, typer.Option(help="Path to merges.txt")],
    vocab_size: Annotated[int, typer.Option(help="Vocabulary size")],
    prompt: Annotated[str, typer.Option(help="Text prompt")] = "Once upon a time",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 256,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.8,
    top_p: Annotated[float, typer.Option(help="Nucleus sampling threshold")] = 0.9,
    context_length: Annotated[int, typer.Option(help="Context length")] = 256,
    d_model: Annotated[int, typer.Option(help="Model dimension")] = 512,
    num_layers: Annotated[int, typer.Option(help="Number of layers")] = 4,
    num_heads: Annotated[int, typer.Option(help="Number of attention heads")] = 16,
    d_ff: Annotated[int, typer.Option(help="Feed-forward dimension")] = 1344,
):
    """Generate text from a trained language model."""
    
    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_file,
        merges_filepath=merges_file,
        special_tokens=["<|endoftext|>"]
    )
    
    # Get EOS token ID
    eos_token_id = None
    eos_bytes = "<|endoftext|>".encode("utf-8")
    for token_id, token_bytes in tokenizer.vocab.items():
        if token_bytes == eos_bytes:
            eos_token_id = token_id
            break
    
    # Load model
    print(f"Loading model from {checkpoint}...")
    ckpt = torch.load(checkpoint, map_location='cpu')
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Encode prompt
    print(f"\nPrompt: {prompt}")
    prompt_tokens = torch.tensor(tokenizer.encode(prompt))
    print(f"Encoded to {len(prompt_tokens)} tokens")
    
    # Generate
    print(f"\nGenerating (max {max_tokens} tokens, temp={temperature}, top_p={top_p})...")
    generated_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    
    # Decode
    generated_text = tokenizer.decode(generated_tokens.tolist())
    
    print("\n" + "=" * 80)
    print("Generated Text:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    print(f"\nGenerated {len(generated_tokens) - len(prompt_tokens)} new tokens")


if __name__ == "__main__":
    app()
