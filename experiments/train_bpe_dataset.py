from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cs336_basics.train_bpe import train_bpe


def _serialize_vocab_merges(
	vocab: dict[int, bytes],
	merges: list[tuple[bytes, bytes]],
	output_dir: Path,
	prefix: str,
) -> tuple[Path, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)

	vocab_path = output_dir / f"{prefix}_vocab.json"
	merges_path = output_dir / f"{prefix}_merges.txt"

	vocab_payload = {str(token_id): token.hex() for token_id, token in vocab.items()}
	with open(vocab_path, "w", encoding="utf-8") as f:
		json.dump(vocab_payload, f, ensure_ascii=False, indent=2)

	with open(merges_path, "w", encoding="utf-8") as f:
		for token_a, token_b in merges:
			f.write(f"{token_a.hex()} {token_b.hex()}\n")

	return vocab_path, merges_path


def _longest_token(vocab: dict[int, bytes]) -> bytes:
	if not vocab:
		return b""
	return max(vocab.values(), key=len)


def _format_token(token: bytes) -> str:
	try:
		decoded = token.decode("utf-8")
	except UnicodeDecodeError:
		decoded = token.decode("latin-1", errors="replace")
	return decoded


def train_bpe_dataset(
	input_path: str | Path,
	vocab_size: int,
	special_tokens: list[str],
	output_dir: str | Path,
	*,
	prefix: str,
	num_processes: int | None = None,
	show_progress: bool = True,
) -> dict[str, Any]:
	"""Train BPE on a dataset and serialize vocab/merges for inspection."""
	vocab, merges, stats = train_bpe(
		input_path=input_path,
		vocab_size=vocab_size,
		special_tokens=special_tokens,
		num_processes=num_processes,
		show_progress=show_progress,
		return_stats=True,
	)

	vocab_path, merges_path = _serialize_vocab_merges(
		vocab=vocab,
		merges=merges,
		output_dir=Path(output_dir),
		prefix=prefix,
	)

	longest = _longest_token(vocab)
	summary = {
		"vocab_path": str(vocab_path),
		"merges_path": str(merges_path),
		"vocab_size": len(vocab),
		"merges_count": len(merges),
		"longest_token_bytes": len(longest),
		"longest_token_preview": _format_token(longest),
		"total_hours": stats["total_s"] / 3600.0,
		"peak_memory_mb": stats["peak_memory_mb"],
		"timings": stats,
	}

	return summary


def train_bpe_tinystories(output_dir: str | Path = "artifacts") -> dict[str, Any]:
	data_path = Path("data") / "TinyStoriesV2-GPT4-train.txt"
	return train_bpe_dataset(
		input_path=data_path,
		vocab_size=10_000,
		special_tokens=["<|endoftext|>"],
		output_dir=output_dir,
		prefix="tinystories",
	)


def train_bpe_expts_owt(output_dir: str | Path = "artifacts") -> dict[str, Any]:
	data_path = Path("data") / "owt_train.txt"
	return train_bpe_dataset(
		input_path=data_path,
		vocab_size=32_000,
		special_tokens=["<|endoftext|>"],
		output_dir=output_dir,
		prefix="owt",
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train BPE tokenizer on a dataset.")
	parser.add_argument(
		"--dataset",
		choices=["tinystories", "owt"],
		required=True,
		help="Dataset to train on: tinystories or owt",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="artifacts",
		help="Output directory for vocab and merges files",
	)

	args = parser.parse_args()

	if args.dataset == "tinystories":
		summary = train_bpe_tinystories(output_dir=args.output_dir)
		print("TinyStories summary:")
	elif args.dataset == "owt":
		summary = train_bpe_expts_owt(output_dir=args.output_dir)
		print("OpenWebText summary:")

	print(json.dumps(summary, indent=2))

