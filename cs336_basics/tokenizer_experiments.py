from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


def sample_documents(filepath: str, n_docs: int = 10, separator: str = "<|endoftext|>") -> list[str]:
    """Sample n random documents from a dataset file.
    
    Args:
        filepath: Path to the dataset file
        n_docs: Number of documents to sample
        separator: Document separator token
        
    Returns:
        List of sampled document strings
    """
    # First pass: count total documents by streaming
    doc_count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            doc_count += line.count(separator)
    
    if doc_count == 0:
        # Fallback: treat entire file as one document
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return [content] if content.strip() else []
    
    # Sample random document indices
    if doc_count < n_docs:
        print(f"Warning: Only {doc_count} documents available, sampling all")
        sample_indices = set(range(doc_count))
    else:
        sample_indices = set(random.sample(range(doc_count), n_docs))
    
    # Second pass: collect only the sampled documents
    sampled_docs = []
    current_doc_idx = 0
    current_doc = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if separator in line:
                parts = line.split(separator)
                for i, part in enumerate(parts):
                    current_doc.append(part)
                    
                    # If not the last part, we've completed a document
                    if i < len(parts) - 1:
                        if current_doc_idx in sample_indices:
                            doc_text = "".join(current_doc).strip()
                            if doc_text:
                                sampled_docs.append(doc_text)
                        
                        current_doc_idx += 1
                        current_doc = []
                        
                        # Early exit if we've collected all samples
                        if len(sampled_docs) == n_docs:
                            return sampled_docs
            else:
                current_doc.append(line)
    
    # Handle last document if it's in our sample
    if current_doc and current_doc_idx in sample_indices:
        doc_text = "".join(current_doc).strip()
        if doc_text:
            sampled_docs.append(doc_text)
    
    return sampled_docs


def calculate_compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> tuple[float, int, int]:
    """Calculate the compression ratio (bytes/token) for a tokenizer on given documents.
    
    Args:
        tokenizer: The tokenizer to use
        documents: List of document strings
        
    Returns:
        Tuple of (compression_ratio, total_bytes, total_tokens)
    """
    total_bytes = 0
    total_tokens = 0
    
    for doc in tqdm(documents, desc="Tokenizing docs", leave=False):
        doc_bytes = len(doc.encode("utf-8"))
        doc_tokens = len(tokenizer.encode(doc))
        
        total_bytes += doc_bytes
        total_tokens += doc_tokens
    
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    return compression_ratio, total_bytes, total_tokens


def measure_throughput(tokenizer: Tokenizer, documents: list[str], n_runs: int = 3) -> float:
    """Measure tokenizer throughput in bytes/second.
    
    Args:
        tokenizer: The tokenizer to use
        documents: List of document strings to tokenize
        n_runs: Number of runs to average over
        
    Returns:
        Throughput in bytes/second
    """
    # Concatenate all documents
    text = " ".join(documents)
    text_bytes = len(text.encode("utf-8"))
    
    times = []
    for _ in tqdm(range(n_runs), desc="Measuring throughput", leave=False):
        start = time.perf_counter()
        _ = tokenizer.encode(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    throughput = text_bytes / avg_time
    
    return throughput


def encode_dataset(
    input_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    separator: str = "<|endoftext|>",
    show_progress: bool = True,
    chunk_size: int = 10_000_000,  # Write to disk every 10M tokens
) -> None:
    """Encode an entire dataset and save as uint16 numpy array.
    
    Processes the file line-by-line and writes chunks to disk to avoid memory issues.
    
    Args:
        input_path: Path to input text file
        output_path: Path to save encoded tokens (as .npy file)
        vocab_path: Path to tokenizer vocabulary file
        merges_path: Path to tokenizer merges file
        special_tokens: List of special tokens
        separator: Document separator
        show_progress: Whether to show progress
        chunk_size: Number of tokens to accumulate before writing to temp file
    """
    print(f"Encoding {input_path}...")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    
    # Get file size for progress bar
    file_size = os.path.getsize(input_path)
    
    # Temporary directory for chunks
    temp_dir = output_path + ".tmp_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Collect tokens in buffer, write chunks to disk when buffer fills
        token_buffer = []
        chunk_files = []
        doc_count = 0
        current_doc = []
        total_tokens = 0
        
        # Create progress bar based on file size
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding") if show_progress else None
        
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if pbar:
                    pbar.update(len(line.encode('utf-8')))
                
                # Check if line contains separator
                if separator in line:
                    # Split by separator
                    parts = line.split(separator)
                    for i, part in enumerate(parts):
                        current_doc.append(part)
                        
                        # If not the last part, we've completed a document
                        if i < len(parts) - 1:
                            doc_text = "".join(current_doc).strip()
                            if doc_text:
                                tokens = tokenizer.encode(doc_text)
                                token_buffer.extend(tokens)
                                # Add separator token
                                separator_tokens = tokenizer.encode(separator)
                                if separator_tokens:
                                    token_buffer.extend(separator_tokens)
                            
                            doc_count += 1
                            current_doc = []
                            
                            # Write chunk to disk if buffer is large enough
                            if len(token_buffer) >= chunk_size:
                                chunk_file = os.path.join(temp_dir, f"chunk_{len(chunk_files):06d}.npy")
                                np.save(chunk_file, np.array(token_buffer, dtype=np.uint16))
                                chunk_files.append(chunk_file)
                                total_tokens += len(token_buffer)
                                token_buffer = []
                                
                                if pbar:
                                    pbar.set_postfix({'docs': doc_count, 'tokens': f'{total_tokens:,}', 'chunks': len(chunk_files)})
                else:
                    current_doc.append(line)
        
        if pbar:
            pbar.close()
        
        # Process any remaining document
        if current_doc:
            doc_text = "".join(current_doc).strip()
            if doc_text:
                tokens = tokenizer.encode(doc_text)
                token_buffer.extend(tokens)
                doc_count += 1
        
        # Write remaining tokens to final chunk
        if token_buffer:
            chunk_file = os.path.join(temp_dir, f"chunk_{len(chunk_files):06d}.npy")
            np.save(chunk_file, np.array(token_buffer, dtype=np.uint16))
            chunk_files.append(chunk_file)
            total_tokens += len(token_buffer)
        
        # Now concatenate all chunks
        if chunk_files:
            print(f"  Merging {len(chunk_files)} chunk files...")
            chunks = []
            for chunk_file in tqdm(chunk_files, desc="Loading chunks", disable=not show_progress):
                chunks.append(np.load(chunk_file))
            
            print(f"  Concatenating {total_tokens:,} tokens...")
            token_array = np.concatenate(chunks)
            
            print(f"  Saving to {output_path}...")
            np.save(output_path, token_array)
            print(f"Saved {len(token_array):,} tokens to {output_path}")
            print(f"Array dtype: {token_array.dtype}, max value: {token_array.max()}")
        else:
            print("No tokens to save!")
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            print(f"  Cleaned up temporary files")


def main():
    """Run all tokenizer experiments."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    artifacts_dir = Path("artifacts")
    data_dir = Path("data")
    
    tinystories_vocab = artifacts_dir / "tinystories_vocab.json"
    tinystories_merges = artifacts_dir / "tinystories_merges.txt"
    owt_vocab = artifacts_dir / "owt_vocab.json"
    owt_merges = artifacts_dir / "owt_merges.txt"
    
    tinystories_train = data_dir / "TinyStoriesV2-GPT4-train.txt"
    tinystories_valid = data_dir / "TinyStoriesV2-GPT4-valid.txt"
    owt_train = data_dir / "owt_train.txt"
    owt_valid = data_dir / "owt_valid.txt"
    
    # Special token
    special_tokens = ["<|endoftext|>"]
    
    print("=" * 80)
    print("TOKENIZER EXPERIMENTS")
    print("=" * 80)

    # Load TinyStories tokenizer (10K vocab)
    print("\n[1] Loading TinyStories tokenizer...")
    ts_tokenizer = Tokenizer.from_files(
        str(tinystories_vocab),
        str(tinystories_merges),
        special_tokens=special_tokens
    )
    print(f"   Vocabulary size: {len(ts_tokenizer.vocab)}")

    # Load OpenWebText tokenizer
    print("\n[2] Loading OpenWebText tokenizer...")
    owt_tokenizer = Tokenizer.from_files(
        str(owt_vocab),
        str(owt_merges),
        special_tokens=special_tokens
    )
    print(f"   Vocabulary size: {len(owt_tokenizer.vocab)}")

    # =========================================================================
    # (a) Sample 10 documents and calculate compression ratios
    # =========================================================================
    print("\n" + "=" * 80)
    print("(a) Compression Ratios on Native Datasets")
    print("=" * 80)

    # TinyStories
    print("\n[TinyStories Dataset]")
    ts_docs = sample_documents(str(tinystories_train), n_docs=10)
    ts_ratio, ts_bytes, ts_tokens = calculate_compression_ratio(ts_tokenizer, ts_docs)
    print(f"  Sampled {len(ts_docs)} documents")
    print(f"  Total bytes: {ts_bytes:,}")
    print(f"  Total tokens: {ts_tokens:,}")
    print(f"  Compression ratio: {ts_ratio:.3f} bytes/token")

    # OpenWebText
    print("\n[OpenWebText Dataset]")
    owt_docs = sample_documents(str(owt_train), n_docs=10)
    owt_ratio, owt_bytes, owt_tokens = calculate_compression_ratio(owt_tokenizer, owt_docs)
    print(f"  Sampled {len(owt_docs)} documents")
    print(f"  Total bytes: {owt_bytes:,}")
    print(f"  Total tokens: {owt_tokens:,}")
    print(f"  Compression ratio: {owt_ratio:.3f} bytes/token")

    # =========================================================================
    # (b) Cross-tokenization: OWT with TinyStories tokenizer
    # =========================================================================
    print("\n" + "=" * 80)
    print("(b) Cross-Tokenization: OpenWebText with TinyStories Tokenizer")
    print("=" * 80)

    owt_docs_sample = sample_documents(str(owt_train), n_docs=10)
    cross_ratio, cross_bytes, cross_tokens = calculate_compression_ratio(
        ts_tokenizer, owt_docs_sample
    )
    print(f"\n  Using TinyStories tokenizer on OpenWebText:")
    print(f"  Total bytes: {cross_bytes:,}")
    print(f"  Total tokens: {cross_tokens:,}")
    print(f"  Compression ratio: {cross_ratio:.3f} bytes/token")

    native_ratio, _, _ = calculate_compression_ratio(owt_tokenizer, owt_docs_sample)
    print(f"\n  Comparison:")
    print(f"    OWT tokenizer on OWT: {native_ratio:.3f} bytes/token")
    print(f"    TS tokenizer on OWT:  {cross_ratio:.3f} bytes/token")
    print(f"    Difference: {((cross_ratio / native_ratio - 1) * 100):.1f}% worse compression")

    # Qualitative example
    print(f"\n  Example tokenization (first 100 chars of first doc):")
    sample_text = owt_docs_sample[0][:100]
    print(f"  Text: {sample_text}")
    tokens = ts_tokenizer.encode(sample_text)
    print(f"  Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    decoded_tokens = [ts_tokenizer.decode([t]) for t in tokens[:20]]
    print(f"  Decoded: {decoded_tokens}")

    # =========================================================================
    # (c) Throughput estimation
    # =========================================================================
    print("\n" + "=" * 80)
    print("(c) Tokenizer Throughput")
    print("=" * 80)

    # Pile dataset size
    pile_size_bytes = 825 * 1024 * 1024 * 1024  # 825 GB in bytes

    # Use TinyStories samples for throughput measurement
    print("\n[Measuring TinyStories tokenizer throughput]")
    ts_throughput = measure_throughput(ts_tokenizer, ts_docs, n_runs=5)
    print(f"  Throughput: {ts_throughput:,.0f} bytes/second ({ts_throughput / 1e6:.2f} MB/s)")

    # Estimate time for Pile dataset (825 GB)
    ts_estimated_seconds = pile_size_bytes / ts_throughput
    ts_estimated_hours = ts_estimated_seconds / 3600
    ts_estimated_days = ts_estimated_hours / 24

    print(f"\n  Estimated time to tokenize Pile dataset (825 GB) with TinyStories tokenizer:")
    print(f"    {ts_estimated_seconds:,.0f} seconds")
    print(f"    {ts_estimated_hours:,.1f} hours")
    print(f"    {ts_estimated_days:.2f} days")

    print("\n[Measuring OpenWebText tokenizer throughput]")
    owt_docs_sample = sample_documents(str(owt_train), n_docs=10)
    owt_throughput = measure_throughput(owt_tokenizer, owt_docs_sample, n_runs=5)
    print(f"  Throughput: {owt_throughput:,.0f} bytes/second ({owt_throughput / 1e6:.2f} MB/s)")

    # Estimate time for Pile dataset (825 GB)
    owt_estimated_seconds = pile_size_bytes / owt_throughput
    owt_estimated_hours = owt_estimated_seconds / 3600
    owt_estimated_days = owt_estimated_hours / 24

    print(f"\n  Estimated time to tokenize Pile dataset (825 GB) with OpenWebText tokenizer:")
    print(f"    {owt_estimated_seconds:,.0f} seconds")
    print(f"    {owt_estimated_hours:,.1f} hours")
    print(f"    {owt_estimated_days:.2f} days")
    
    # =========================================================================
    # (d) Encode datasets to numpy arrays
    # =========================================================================
    print("\n" + "=" * 80)
    print("(d) Encoding Datasets to Token IDs")
    print("=" * 80)
    
    print("\nWhy uint16?")
    print("  - TinyStories vocab size: ~10K tokens (max ~10,000)")
    print("  - OpenWebText vocab size: ~32K tokens (max ~32,000)")
    print("  - uint16 range: 0 to 65,535")
    print("  - uint16 is sufficient to represent all token IDs")
    print("  - Uses 2 bytes per token (vs 4 for uint32 or 8 for int64)")
    print("  - Saves significant storage space for large datasets\n")

    # Encode TinyStories datasets
    print("\n[Encoding TinyStories datasets]")
    encode_dataset(
        str(tinystories_train),
        str(artifacts_dir / "tinystories_train_encoded.npy"),
        str(tinystories_vocab),
        str(tinystories_merges),
        special_tokens,
        separator="<|endoftext|>",
    )

    encode_dataset(
        str(tinystories_valid),
        str(artifacts_dir / "tinystories_valid_encoded.npy"),
        str(tinystories_vocab),
        str(tinystories_merges),
        special_tokens,
        separator="<|endoftext|>",
    )
    
    # Encode OpenWebText datasets
    print("\n[Encoding OpenWebText datasets]")
    encode_dataset(
        str(owt_train),
        str(artifacts_dir / "owt_train_encoded.npy"),
        str(owt_vocab),
        str(owt_merges),
        special_tokens,
        separator="<|endoftext|>",
    )
    
    encode_dataset(
        str(owt_valid),
        str(artifacts_dir / "owt_valid_encoded.npy"),
        str(owt_vocab),
        str(owt_merges),
        special_tokens,
        separator="<|endoftext|>",
    )
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
