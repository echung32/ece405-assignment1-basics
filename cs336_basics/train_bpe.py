from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from typing import BinaryIO

import multiprocessing as mp
import psutil
import regex as re
from tqdm import tqdm


# GPT-2 pre-tokenizer
_GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _process_chunk(path: str, start: int, end: int, pattern: str) -> dict[tuple[bytes, ...], int]:
    frequency: dict[tuple[bytes, ...], int] = Counter()

    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # Run pre-tokenization on your chunk and store the counts for each pre-token
    documents = re.split(pattern, text)
    # each batch has multiple documents, split by <|endoftext|> for many documents
    for doc in documents:
        # run tokenizer on each document individually
        for m in re.finditer(_GPT2_PAT, doc):
            tok = m.group(0)
            tok_bytes = tok.encode("utf-8")
            # It is convenient to represent this as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}.
            tok_tuple = tuple(bytes([b]) for b in tok_bytes)
            frequency[tok_tuple] += 1

    return frequency


def _pretokenize(path: str, num_processes: int, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    import multiprocessing as mp
    
    with open(path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8") if special_tokens else b"\n")

    # escapes the | in the special tokens
    pattern = "|".join([re.escape(tok) for tok in special_tokens])

    # create chunks and run pre-tokenization
    tasks = [(path, start, end, pattern) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with mp.Pool(processes=num_processes) as pool:
        result = pool.starmap(_process_chunk, tasks)

        # combine the individual counters
        total_frequency: dict[tuple[bytes, ...], int] = Counter()
        for freq in result:
            total_frequency.update(freq)

    return total_frequency



def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int | None = None,
    show_progress: bool = True,
    log_timings: bool = True,
    return_stats: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] | tuple[
    dict[int, bytes], list[tuple[bytes, bytes]], dict[str, float]
]:
    train_start = time.perf_counter()

    process = psutil.Process(os.getpid())

    if num_processes is None:
        num_processes = min(mp.cpu_count(), 16)
        print(f"Using num_processes={num_processes}")

    pretokenize_start = time.perf_counter()
    frequency = _pretokenize(str(input_path), num_processes, special_tokens)
    pretokenize_elapsed = time.perf_counter() - pretokenize_start
    if log_timings:
        print(f"{len(frequency)} unique pre-tokens found in the dataset.")

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {}
    vocab_idx = 0

    vocab_start = time.perf_counter()
    # first init with the special tokens
    for tok in special_tokens:
        vocab[vocab_idx] = tok.encode("utf-8")
        vocab_idx += 1

    # then add our initial vocab list
    # these represent the initial range from 0-255
    # https://www.ascii-code.com/
    for b in range(256):
        vocab[vocab_idx] = bytes([b])
        vocab_idx += 1
    vocab_elapsed = time.perf_counter() - vocab_start

    # keep track of the pairs (like l,o ; o,w and the counts)
    pair_count_start = time.perf_counter()
    pair_counts: dict[tuple[bytes, bytes], int] = Counter()
    # keep reverse mapping of which tokens contain a given pair, to speed up updates after merges
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for tok, count in frequency.items():
        if len(tok) == 1:
            continue # nothing to merge
        for b1, b2 in zip(tok[:-1], tok[1:]):
            pair_counts[(b1, b2)] += count
            pair_to_tokens[(b1, b2)].add(tok)
    pair_count_elapsed = time.perf_counter() - pair_count_start

    merge_start = time.perf_counter()
    merge_update_elapsed = 0.0

    total_merges = max(vocab_size - len(vocab), 0)
    progress_context = tqdm(total=total_merges, desc="Merging") if show_progress else nullcontext()
    with progress_context as pbar:
        while len(vocab) < vocab_size:
            if not pair_counts:
                # no more pairs to merge
                break

            # now we are processing merges, find the most frequent pair
            # take lexicographical max to break ties, x[1] is value, then x[0] is key
            most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

            vocab[vocab_idx] = most_frequent_pair[0] + most_frequent_pair[1]
            vocab_idx += 1
            merges.append(most_frequent_pair)

            update_start = time.perf_counter()
            affected_tokens = list(pair_to_tokens.get(most_frequent_pair, set()))
            for tok in affected_tokens:
                count = frequency.get(tok, 0)
                if count == 0:
                    continue
                # maintain list of new token components as we iterate through the old token, merging pairs as we go
                _new_tok: list[bytes] = []
                idx = 0
                merged = False

                while idx < len(tok):
                    # build new token by going thru all pairs first (vs. editing while going thru)
                    if idx < len(tok) - 1 and (tok[idx], tok[idx + 1]) == most_frequent_pair:
                        _new_tok.append(tok[idx] + tok[idx + 1])
                        merged = True
                        idx += 2 # skip 2 bc merged pair
                    else:
                        _new_tok.append(tok[idx])
                        idx += 1

                new_tok: tuple[bytes, ...] = tuple(_new_tok)
                if merged:
                    # todo: can do in one for loop later but this was just a bit more intuitive
                    tok_pairs = [(tok[i], tok[i + 1]) for i in range(len(tok) - 1)]
                    new_tok_pairs = [(new_tok[i], new_tok[i + 1]) for i in range(len(new_tok) - 1)]

                    for old_pair in tok_pairs:
                        pair_counts[old_pair] -= count
                        if pair_counts[old_pair] <= 0:
                            del pair_counts[old_pair]
                        pair_to_tokens[old_pair].discard(tok)
                        if not pair_to_tokens[old_pair]:
                            del pair_to_tokens[old_pair]

                    for new_pair in new_tok_pairs:
                        pair_counts[new_pair] += count
                        pair_to_tokens[new_pair].add(new_tok)

                    frequency[tok] -= count
                    if frequency[tok] <= 0:
                        del frequency[tok]
                    frequency[new_tok] = frequency.get(new_tok, 0) + count
            merge_update_elapsed += time.perf_counter() - update_start

            if show_progress:
                pbar.update(1)

    merge_elapsed = time.perf_counter() - merge_start

    merge_update_avg = merge_update_elapsed / len(merges) if merges else 0.0

    peak_memory = process.memory_info().rss / (1024 ** 2)
    total_elapsed = time.perf_counter() - train_start

    stats = {
        "pretokenize_s": pretokenize_elapsed,
        "vocab_init_s": vocab_elapsed,
        "pair_count_init_s": pair_count_elapsed,
        "merge_loop_total_s": merge_elapsed,
        "merge_updates_total_s": merge_update_elapsed,
        "merge_update_avg_s": merge_update_avg,
        "total_s": total_elapsed,
        "peak_memory_mb": peak_memory,
    }

    if log_timings:
        print(
            "Timing (s): "
            f"pretokenize={pretokenize_elapsed:.3f}, vocab_init={vocab_elapsed:.3f}, "
            f"pair_count_init={pair_count_elapsed:.3f}, merge_loop_total={merge_elapsed:.3f}, "
            f"merge_updates_total={merge_update_elapsed:.3f}, merge_update_avg={merge_update_avg:.6f}, "
            f"total={total_elapsed:.3f}"
        )
        print(f"Peak Memory Usage: {peak_memory:.2f} MB")

    if return_stats:
        return vocab, merges, stats

    return vocab, merges
