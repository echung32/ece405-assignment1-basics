from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import regex as re


# GPT-2 pre-tokenizer pattern
_GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """A BPE tokenizer that encodes text into token IDs and decodes token IDs back to text."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab: dict[int, bytes] - Mapping from token ID to token bytes
            merges: list[tuple[bytes, bytes]] - List of BPE merges in order
            special_tokens: list[str] | None - Optional list of special tokens
        """
        self.vocab = vocab.copy()
        self.special_tokens = special_tokens or []

        # Add special tokens to vocab if they aren't already there
        next_id = max(vocab.keys()) + 1 if vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in vocab.values():
                self.vocab[next_id] = token_bytes
                next_id += 1

        # Create reverse mapping: bytes -> token ID
        self.byte_to_id = {v: k for k, v in self.vocab.items()}

        # Build merge ranking: (byte1, byte2) -> merge_rank
        # Lower rank means higher priority (merged earlier)
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # Compile regex pattern for special tokens (to split on them)
        if self.special_tokens:
            # Sort special tokens by length (descending) to match longer tokens first
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # Escape special tokens and join with |
            special_pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
            self.special_pattern = re.compile(f"({special_pattern})")
        else:
            self.special_pattern = None

        # Compile GPT-2 pre-tokenization pattern
        self.gpt2_pattern = re.compile(_GPT2_PAT)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Class method that constructs and returns a Tokenizer from serialized vocabulary and list of merges.

        Args:
            vocab_filepath: str - Path to vocabulary JSON file
            merges_filepath: str - Path to merges text file
            special_tokens: list[str] | None - Optional list of special tokens

        Returns:
            Tokenizer instance
        """
        # Load vocabulary
        with open(vocab_filepath, "r") as f:
            vocab_json = json.load(f)

        # Convert from JSON format (string keys, hex string values) to dict[int, bytes]
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_json.items()}

        # Load merges
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        # Convert hex strings to bytes
                        byte1 = bytes.fromhex(parts[0])
                        byte2 = bytes.fromhex(parts[1])
                        merges.append((byte1, byte2))

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, token_bytes: bytes) -> list[bytes]:
        """Apply BPE merges to a byte sequence.

        Args:
            token_bytes: bytes - The byte sequence to apply merges to

        Returns:
            list[bytes] - List of byte sequences after applying merges
        """
        # Start with individual bytes
        tokens = [bytes([b]) for b in token_bytes]

        if len(tokens) < 2:
            return tokens

        # Keep merging until no more merges can be applied
        while True:
            # Find the pair with the lowest merge rank (highest priority)
            best_pair = None
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                        best_idx = i

            # If no merge found, we're done
            if best_pair is None:
                break

            # Merge the best pair
            merged = best_pair[0] + best_pair[1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2 :]

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            text: str - The text to encode

        Returns:
            list[int] - List of token IDs
        """
        if not text:
            return []

        token_ids = []

        # Split on special tokens first if they exist
        if self.special_pattern:
            chunks = self.special_pattern.split(text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue

            # Check if this chunk is a special token
            if chunk in self.special_tokens:
                token_bytes = chunk.encode("utf-8")
                token_ids.append(self.byte_to_id[token_bytes])
            else:
                # Apply GPT-2 pre-tokenization to split into words/punctuation
                for match in self.gpt2_pattern.finditer(chunk):
                    pre_token = match.group(0)
                    pre_token_bytes = pre_token.encode("utf-8")

                    # Apply BPE merges
                    merged_tokens = self._apply_merges(pre_token_bytes)

                    # Convert to token IDs
                    for token in merged_tokens:
                        token_ids.append(self.byte_to_id[token])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings, return a generator that lazily yields token IDs.

        Args:
            iterable: Iterable[str] - An iterable of strings (e.g., a file handle)

        Yields:
            int - Token IDs
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids: list[int] - List of token IDs

        Returns:
            str - Decoded text
        """
        # Concatenate all byte sequences
        byte_sequence = b"".join(self.vocab.get(token_id, b"") for token_id in ids)

        # Decode to Unicode, replacing malformed bytes with U+FFFD
        return byte_sequence.decode("utf-8", errors="replace")
