import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import typer
import wandb
from tqdm import tqdm

from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.optimizer import AdamW
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.transformer_lm import TransformerLM

app = typer.Typer()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.command()
def train(
        vocab_size: int = typer.Option(..., help="Size of the vocabulary"),
        train_data: str = typer.Option(..., help="Path to training data (.npy file)"),
        val_data: str = typer.Option(..., help="Path to validation data (.npy file)"),
        device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device to use"),

        # Model hyperparameters (as provided in 7.2)
        context_length: int = typer.Option(256, help="Maximum sequence length"),
        d_model: int = typer.Option(512, help="Model dimension"),
        num_layers: int = typer.Option(4, help="Number of transformer layers"),
        num_heads: int = typer.Option(16, help="Number of attention heads"),
        d_ff: int = typer.Option(1344, help="Feed-forward dimension"),
        rope_theta: float = typer.Option(10000.0, help="RoPE theta parameter"),

        # Optimizer hyperparameters
        learning_rate: float = typer.Option(3e-4, help="Learning rate"),
        weight_decay: float = typer.Option(0.001, help="Weight decay for AdamW"),
        # Typical applications set (β1, β2) to (0.9, 0.999), but large
        # language models like LLaMA [Touvron et al., 2023] and GPT-3 [Brown et al., 2020] are often trained with
        # (0.9, 0.95). So let's use 0.9 and 0.95
        beta1: float = typer.Option(0.9, help="Adam beta1"),
        beta2: float = typer.Option(0.95, help="Adam beta2"),
        eps: float = typer.Option(1e-8, help="Adam epsilon"),

        # Learning rate schedule
        max_lr: float = typer.Option(6e-4, help="Maximum learning rate for schedule"),
        min_lr: float = typer.Option(0.0, help="Minimum learning rate for schedule"),
        warmup_iters: int = typer.Option(2000, help="Number of warmup iterations"),
        # "When using X training steps, we suggest adjusting the cosine learning rate decay
        # schedule to terminate its decay (i.e., reach the minimum learning rate) at precise step X."
        # So this should just match max_iters instead.
        # lr_decay_iters: int = typer.Option(40000, help="Number of iterations for LR decay"),

        # Training hyperparameters
        batch_size: int = typer.Option(32, help="Batch size"),
        max_iters: int = typer.Option(40000, help="Maximum number of training iterations"),
        grad_clip: float = typer.Option(1.0, help="Gradient clipping threshold"),

        # Checkpointing and logging
        checkpoint_dir: str = typer.Option("./checkpoints", help="Directory to save checkpoints"),
        resume_from: Optional[str] = typer.Option(None, help="Path to checkpoint to resume from"),
        save_interval: int = typer.Option(5000, help="Save checkpoint every N iterations"),
        log_interval: int = typer.Option(100, help="Log training metrics every N iterations"),
        eval_interval: int = typer.Option(500, help="Evaluate every N iterations"),
        eval_iters: int = typer.Option(200, help="Number of batches for evaluation"),

        # Other settings
        seed: int = typer.Option(42, help="Random seed"),
        wandb_entity: str = typer.Option("echung32-ece405", help="Weights & Biases entity name"),
        wandb_project: str = typer.Option("assignment-1", help="Weights & Biases project name"),
        wandb_run_name: str = typer.Option(None, help="Weights & Biases run name"),
):
    """Train a Transformer Language Model."""

    # Print configuration
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Model:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Context length: {context_length}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  RoPE theta: {rope_theta}")
    print(f"\nOptimizer:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Beta1: {beta1}")
    print(f"  Beta2: {beta2}")
    print(f"  Epsilon: {eps}")
    print(f"\nLearning Rate Schedule:")
    print(f"  Max LR: {max_lr}")
    print(f"  Min LR: {min_lr}")
    print(f"  Warmup iterations: {warmup_iters:,}")
    print(f"\nTraining:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max iterations: {max_iters:,}")
    print(f"  Gradient clipping: {grad_clip}")
    print(f"\nData:")
    print(f"  Training data: {train_data}")
    print(f"  Validation data: {val_data}")
    print(f"\nCheckpointing:")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Save interval: {save_interval:,} iterations")
    print(f"  Eval interval: {eval_interval:,} iterations")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    print(f"\nDevice: {device}")
    print(f"Random seed: {seed}")
    print(f"Weights & Biases: enabled (project: {wandb_project})")
    print("=" * 80)

    # Set random seed
    set_seed(seed)

    # https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30668820/Deep+Learning+on+A100+GPUs
    # https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    # since we are training in fp32 but a100 has 2x fp16 flops, let it use them.
    # high mode = either tf32 or float32 number as the sum of two bfloat16 numbers
    torch.set_float32_matmul_precision('high')

    # Initialize Weights & Biases
    wandb_run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=wandb_run_name,
        config={
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_iters": warmup_iters,
            "batch_size": batch_size,
            "max_iters": max_iters,
            "grad_clip": grad_clip,
            "seed": seed,
        }
    )

    # Create a checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # use memmap to load data
    #  since we don't have a dataloader implemented, the gpu is idle when it waits for new batches.
    #  so just use more ram in exchange to load data fully into memory.
    # train_mmap = np.memmap(train_data, dtype=np.uint16, mode='r')
    # val_mmap = np.memmap(val_data, dtype=np.uint16, mode='r')

    # we should also be able to train both directly in memory.
    # $ ls -lh artifacts/*encoded*.npy
    # 5.1G Feb 11 23:01 artifacts/owt_train_encoded.npy
    # 127M Feb 11 23:04 artifacts/owt_valid_encoded.npy
    # 1021M Feb 11 20:01 artifacts/tinystories_train_encoded.npy
    # 11M Feb 11 20:02 artifacts/tinystories_valid_encoded.npy

    # directly load data into ram
    train_dataset = np.load(train_data)
    val_dataset = np.load(val_data)

    # Initialize model
    print("\nInitializing model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")

    # https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
    # speedups using a100 should be significant, the first batch will take longer though.
    model = torch.compile(model, mode="default")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )

    # Resume from a checkpoint if provided
    start_iter = 0
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        start_iter = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    model.train()
    train_losses = []
    start_time = time.time()

    pbar = tqdm(range(start_iter, max_iters), initial=start_iter, total=max_iters, desc="Training")

    for iteration in pbar:
        # Update learning rate
        lr = get_lr_cosine_schedule(
            iteration=iteration,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        inputs, targets = get_batch(
            train_dataset,
            batch_size,
            context_length,
            device
        )

        # Forward pass
        logits = model(inputs)

        # Compute loss
        batch_size_actual, seq_len, vocab_size_actual = logits.shape
        logits_flat = logits.view(batch_size_actual * seq_len, vocab_size_actual)
        targets_flat = targets.view(batch_size_actual * seq_len)
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            gradient_clipping(model.parameters(), grad_clip)

        # Optimizer step
        optimizer.step()

        train_losses.append(loss.item())

        # Update progress bar
        avg_loss = np.mean(train_losses[-log_interval:]) if len(train_losses) >= log_interval else np.mean(train_losses)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})

        # Logging
        if iteration % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_size * context_length * log_interval) / elapsed if elapsed > 0 else 0

            print(f"\nIter {iteration:6d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                  f"Tokens/sec: {tokens_per_sec:.0f}")

            wandb.log({
                "train/loss": avg_loss,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_sec,
                "iteration": iteration
            })

            start_time = time.time()

        # Evaluation
        if iteration % eval_interval == 0 and iteration > 0:
            print(f"\nEvaluating at iteration {iteration}...")
            results = evaluate(
                model, train_dataset, val_dataset,
                batch_size, context_length, device, eval_iters
            )
            print(f"Iter {iteration:6d} | "
                  f"Train loss: {results['train']['loss']:.4f} (ppl: {results['train']['perplexity']:.2f}) | "
                  f"Val loss: {results['val']['loss']:.4f} (ppl: {results['val']['perplexity']:.2f})")

            wandb.log({
                "eval/train_loss": results['train']['loss'],
                "eval/train_perplexity": results['train']['perplexity'],
                "eval/val_loss": results['val']['loss'],
                "eval/val_perplexity": results['val']['perplexity'],
                "iteration": iteration
            })

        # Save checkpoint
        if iteration % save_interval == 0 and iteration > 0:
            checkpoint_file = checkpoint_path / f"checkpoint_iter_{iteration}.pt"
            print(f"\nSaving checkpoint to {checkpoint_file}")
            save_checkpoint(model, optimizer, iteration, checkpoint_file)

            # Also save as "latest" for easy resumption
            latest_file = checkpoint_path / "checkpoint_latest.pt"
            save_checkpoint(model, optimizer, iteration, latest_file)

    pbar.close()

    # Final checkpoint
    final_checkpoint_file = checkpoint_path / "checkpoint_final.pt"
    print(f"\nSaving final checkpoint to {final_checkpoint_file}")
    save_checkpoint(model, optimizer, max_iters, final_checkpoint_file)

    # Final evaluation
    print("\nFinal evaluation...")
    results = evaluate(
        model, train_dataset, val_dataset,
        batch_size, context_length, device, eval_iters
    )
    print(f"Final | "
          f"Train loss: {results['train']['loss']:.4f} (ppl: {results['train']['perplexity']:.2f}) | "
          f"Val loss: {results['val']['loss']:.4f} (ppl: {results['val']['perplexity']:.2f})")

    wandb.log({
        "final/train_loss": results['train']['loss'],
        "final/train_perplexity": results['train']['perplexity'],
        "final/val_loss": results['val']['loss'],
        "final/val_perplexity": results['val']['perplexity'],
    })
    wandb_run.finish()


@torch.no_grad()
def evaluate(model: nn.Module, train_data, val_data, batch_size: int,
             context_length: int, device: str, eval_iters: int):
    """
    Evaluate model on training and validation sets.

    Args:
        model: The model to evaluate
        train_data: Training dataset
        val_data: Validation dataset
        batch_size: Batch size for evaluation
        context_length: Context length
        device: Device to use
        eval_iters: Number of batches to evaluate

    Returns:
        Dictionary with 'train' and 'val' loss and perplexity values
    """
    model.eval()
    results = {}

    for split, data in [('train', train_data), ('val', val_data)]:
        losses_list = []

        eval_pbar = tqdm(range(eval_iters), desc=f"Evaluating {split}", leave=False)
        for _ in eval_pbar:
            inputs, targets = get_batch(
                data,
                batch_size,
                context_length,
                device
            )
            logits = model(inputs)

            # Compute loss for each position
            # logits: (batch_size, seq_len, vocab_size)
            # targets: (batch_size, seq_len)
            batch_size_actual, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size_actual * seq_len, vocab_size)
            targets_flat = targets.view(batch_size_actual * seq_len)

            loss = cross_entropy(logits_flat, targets_flat)
            losses_list.append(loss.item())

        avg_loss = np.mean(losses_list)
        perplexity = np.exp(avg_loss)

        results[split] = {
            'loss': avg_loss,
            'perplexity': perplexity
        }

    model.train()
    return results


if __name__ == "__main__":
    app()
