"""
Fix checkpoint files that were saved with torch.compile's _orig_mod. prefix.
This strips the prefix to make the checkpoint loadable into non-compiled models.
"""

import torch
import typer

app = typer.Typer()


@app.command()
def main(
    checkpoint_path: str = typer.Argument(..., help="Path to checkpoint file to fix"),
    output_path: str = typer.Option(None, help="Output path (default: overwrite input)"),
):
    """Remove _orig_mod. prefix from state dict keys in a checkpoint."""
    
    if output_path is None:
        output_path = checkpoint_path
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if this checkpoint needs fixing
    state_dict = checkpoint['model_state_dict']
    sample_key = next(iter(state_dict.keys()))
    
    if not sample_key.startswith('_orig_mod.'):
        print("Checkpoint does not have _orig_mod. prefix. No fix needed.")
        return
    
    print(f"Found {len(state_dict)} keys with _orig_mod. prefix")
    print("Stripping prefix...")
    
    # Create new state dict with stripped keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    checkpoint['model_state_dict'] = new_state_dict
    
    print(f"Saving fixed checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("Done!")


if __name__ == "__main__":
    app()
