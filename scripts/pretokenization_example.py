import os
import regex as re
from typing import BinaryIO, Dict, Counter
import multiprocessing as mp


def find_chunk_boundaries(
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

# GPT-2 pre-tokenizer
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<|endoftext|>"]
# escapes the | in the special tokens
PATTERN = "|".join([re.escape(tok) for tok in SPECIAL_TOKENS])

def process_chunk(path: int, start: int, end: int):
    frequency: dict[tuple[bytes, ...], int] = Counter()

    with (open(path, "rb") as f):
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # Run pre-tokenization on your chunk and store the counts for each pre-token
    documents = re.split(PATTERN, text)
    # each batch has multiple documents, split by <|endoftext|> for many documents
    for doc in documents:
        # run tokenizer on each document individually
        for m in re.finditer(GPT2_PAT, doc):
            tok = m.group(0)
            tok_bytes = tok.encode("utf-8")
            # It is convenient to represent this as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}.
            tok_tuple = tuple(bytes([b]) for b in tok_bytes)
            # print(tok_tuple)
            frequency[tok_tuple] += 1

    return frequency

## Usage
if __name__ == "__main__":

    DATA_PATH = "../data/TinyStoriesV2-GPT4-valid.txt"
    num_processes = 8

    with open(DATA_PATH, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # create chunks and run pre-tokenization
    tasks = [(DATA_PATH, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with mp.Pool(processes=num_processes) as pool:
        result = pool.starmap(process_chunk, tasks)

        # combine the individual counters
        total_frequency: dict[tuple[bytes, ...], int] = Counter()
        for freq in result:
            total_frequency.update(freq)
        # print(total_frequency)

    ##### ---- bpe merging now
    # cannot do the merge parallel, as instructions state.

    vocab_size = 5000
    merges: list[tuple[bytes, bytes]] = []
    vocab: Dict[int, bytes] = {}
    vocab_idx = 0

    # first init with the special tokens
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[vocab_idx] = tok.encode("utf-8")
        vocab_idx += 1

    # then add our initial vocab list
    # these represent the initial range from 0-255
    # https://www.ascii-code.com/
    for b in range(256):
        vocab[vocab_idx] = bytes([b])
        vocab_idx += 1

    # keep track of the pairs (like l,o ; o,w and the counts
    pair_counts: dict[tuple[bytes, bytes], int] = Counter()

    for tok, count in total_frequency.items():
        if len(tok) == 1:
            continue # nothing to merge
        for b1, b2 in zip(tok[:-1], tok[1:]):
            # print(tok, b1, b2)
            pair_counts[(b1, b2)] += count

    while len(vocab) < vocab_size:

        # now we are processing merges, find the most frequent pair
        # take lexicographical max to break ties, x[1] is value, then x[0] is key
        most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        print(pair_counts.most_common(5))
        print(most_frequent_pair)

        # update the vocab now
        vocab[vocab_idx] = most_frequent_pair[0] + most_frequent_pair[1]
        vocab_idx += 1
        merges.append(most_frequent_pair)

        # with the most frequent pair, merge their occurrences in total_frequency
        # need to create copy as a list because modifying dict directly
        for tok, count in list(total_frequency.items()):
            # maintain list of new tokens to create at the end
            _new_tok: list[bytes] = []
            idx = 0
            merged = False

            while idx < len(tok):
                # build new token by going thru all pairs first (vs. editing while going thru)
                if idx < len(tok) - 1 and (tok[idx], tok[idx+1]) == most_frequent_pair:
                    _new_tok.append(tok[idx] + tok[idx+1])
                    merged = True
                    idx += 2 # skip 2 because we merged 2 into 1
                else:
                    # case where there's no match
                    _new_tok.append(tok[idx])
                    idx += 1

            new_tok: tuple[bytes, ...] = tuple(_new_tok)
            # now update the pair counts if there was a merge
            if merged:
                # todo: can do in one for loop later but this was just a bit more intuitive
                tok_pairs = [(tok[i], tok[i+1]) for i in range(len(tok)-1)]
                new_tok_pairs = [(new_tok[i], new_tok[i+1]) for i in range(len(new_tok)-1)]

                for old_pair in tok_pairs:
                    pair_counts[old_pair] -= count
                    if pair_counts[old_pair] <= 0: del pair_counts[old_pair]

                for new_pair in new_tok_pairs:
                    pair_counts[new_pair] += count

                # update the frequency counter
                total_frequency[tok] -= count
                if total_frequency[tok] <= 0: del total_frequency[tok]
                # cant += on something not defined yet, so use get+count instead
                total_frequency[new_tok] = total_frequency.get(new_tok, 0) + count

        # THIS IS BAD IMPLEMENTATION DON'T USE because index shifting messes it up
        # for tok, count in list(total_frequency.items()):
        #     if len(tok) == 1:
        #         continue # nothing to merge
        #     for i, (b1, b2) in enumerate(zip(tok[:-1], tok[1:])):
        #         if (b1, b2) == most_frequent_pair:
        #             # print(i, tok, (b1, b2), tok[:i+2], tok[i+2:])
        #             # take everything before the pair, add the merged pair, and remove 2 from rest of tok.
        #             new_tok = tok[:i] + (b1 + b2,) + tok[i+2:]
        #             total_frequency[new_tok] += count
        #             total_frequency[tok] -= count
        #             # print(new_tok, total_frequency.get(new_tok), total_frequency.get(tok))
        #
        #             # now we need to update the pair counts
        #             pair_counts[(b1, b2)] -= count # decrement the pair we just merged
        #             if pair_counts[(b1, b2)] == 0: del pair_counts[(b1, b2)]
        #
        #             # the adjacent on the left
        #             if i > 0:
        #                 pair_counts[(tok[i-1], b1)] -= count
        #                 if pair_counts[(tok[i-1], b1)] == 0: del pair_counts[(tok[i-1], b1)]
        #                 pair_counts[(new_tok[i - 1], new_tok[i])] += total_frequency[new_tok]
        #
        #             # the adjacent on the right
        #             if i < len(tok) - 1:
        #                 pair_counts[(b2, tok[i+1])] -= count
        #                 if pair_counts[(b2, tok[i+1])] == 0: del pair_counts[(b2, tok[i+1])]
        #                 pair_counts[(new_tok[i], new_tok[i+1])] += total_frequency[new_tok]
        #
        #             # add the new adjacent pairs
        #             # print(new_tok[i-1], new_tok[i], new_tok[i+1])
        #             # print(pair_counts[(new_tok[i-1], new_tok[i])], pair_counts[(new_tok[i], new_tok[i+1])])

    print(vocab)
    print(merges)

# for file, start, end in [(DATA_PATH, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]:
#     print(file, start ,end)

# The following is a serial implementation, but you can parallelize this
# by sending each start/end pair to a set of processes.
# for start, end in zip(boundaries[:-1], boundaries[1:]):
#     f.seek(start)
#     chunk = f.read(end - start).decode("utf-8", errors="ignore")

