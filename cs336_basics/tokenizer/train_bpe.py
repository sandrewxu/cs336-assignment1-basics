import os
import regex as re
from collections import defaultdict
import multiprocessing
import mmap
from typing import List, Tuple, Dict, DefaultDict

GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

# --- PRETOKENIZATION HELPERS ---
def get_chunk_boundaries(filename: str, num_chunks: int, special_token: str = None) -> List[Tuple[int, int]]:
    file_size = os.path.getsize(filename)

    # If no special token, we can't safely chunk - use single chunk
    if not special_token:
        return [(0, file_size)]

    chunk_size = file_size // num_chunks
    boundaries = []
    start = 0
    token_bytes = special_token.encode("utf-8")

    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i in range(num_chunks):
                target_end = min(start + chunk_size, file_size)
                if i == num_chunks - 1:
                    end = file_size
                else:
                    # Search backward first in a reasonable window
                    last_token_pos = mm.rfind(token_bytes, start, target_end)
                    
                    if last_token_pos != -1:
                        # Found special token before target_end
                        end = last_token_pos + len(token_bytes)
                    else:
                        # No special token found before target_end
                        # Search forward from target_end to find next occurrence
                        next_token_pos = mm.find(token_bytes, target_end)
                        
                        if next_token_pos != -1:
                            end = next_token_pos + len(token_bytes)
                        else:
                            # No more special tokens - this becomes the last chunk
                            end = file_size

                if start < end:
                    boundaries.append((start, end))
                start = end

                # If we've reached the end, stop creating chunks
                if start >= file_size:
                    break
    return boundaries

def process_chunk_mmap(args: Tuple) -> Dict[Tuple[int, ...], int]:
    input_path, start, end, special_tokens = args
    word_freqs = defaultdict(int)

    with open(input_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            chunk_bytes = mm[start:end]
            text_chunk = chunk_bytes.decode('utf-8', errors='ignore')

    sub_chunks = [text_chunk]
    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        sub_chunks = re.split(f"({special_pattern})", text_chunk)

    for sc in sub_chunks:
        if not sc or sc in special_tokens:
            continue
        for match in GPT2_PAT.finditer(sc):
            word_bytes = match.group(0).encode("utf-8")
            words_as_ids = tuple(b for b in word_bytes)
            word_freqs[words_as_ids] += 1
    return word_freqs

# --- BPE HELPERS ---
def get_initial_pair_stats(word_freqs: Dict[Tuple[int, ...], int]) -> DefaultDict[Tuple[int, int], int]:
    """Calculates the initial frequency of all adjacent pairs."""
    pair_stats = defaultdict(int)
    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair_stats[(word[i], word[i+1])] += freq
    return pair_stats

def merge_pair_and_update_stats(
    word_freqs: Dict[Tuple[int, ...], int],
    pair_to_merge: Tuple[int, int],
    new_id: int,
    pair_stats: DefaultDict[Tuple[int, int], int]
) -> Dict[Tuple[int, ...], int]:
    """
    Merges a pair and incrementally updates the pair_stats dictionary.
    """
    new_word_freqs = defaultdict(int)
    p1, p2 = pair_to_merge

    for word, freq in word_freqs.items():
        # only one token, cannot be merged
        if len(word) < 2:
            new_word_freqs[word] += freq
            continue
        # multiple tokens: remove from pair_stats, recombine word ot word_freqs, pair_stats
        new_word = []
        merged = False
        i = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i+1]) == (p1, p2):
                merged = True
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if merged:
            for i in range(len(word) - 1):
                pair_stats[word[i], word[i+1]] -= freq
            for i in range(len(new_word) - 1):
                pair_stats[new_word[i], new_word[i+1]] += freq
            new_word_freqs[tuple(new_word)] = freq
        else:
            new_word_freqs[tuple(word)] = freq
    return new_word_freqs

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], verbose=False) -> Tuple[Dict[int, bytes], 
List[Tuple[bytes, bytes]]]:
    """
    given a path to an input text file, train a (byte-level) BPE tokenizer

    implements parallel pre-tokenization and optimized merging

    returns vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
    """
    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least {min_vocab_size} to accomodate all byte tokens and special tokens.")
    
    # 1. Parallel Pre-tokenization
    num_workers = os.cpu_count() or 1
    chunk_delimiter = special_tokens[0] if special_tokens else None

    if verbose:
        print(f"Reading file '{input_path}'...")
        print(f"Chunking file for parallel processing with {num_workers} workers...")
    boundaries = get_chunk_boundaries(input_path, num_workers, chunk_delimiter)
    pool_args = [(input_path, start, end, special_tokens) for start, end in boundaries]

    word_freqs = defaultdict(int)
    if verbose:
        print("Starting parallel pre-tokenization...")
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(process_chunk_mmap, pool_args)
    
    if verbose:
        print("Aggregating results from workers...")
    for res_dict in results:
        for word, freq in res_dict.items():
            word_freqs[word] += freq

    if verbose:
        print(f"Pre-tokenization complete. Found {len(word_freqs)} unique words.")

    # 2. Optimized BPE algorithm
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        vocab[256 + i] = token_str.encode('utf-8')
    
    if verbose:
        print("Calculating initial pair statistics...")
    pair_stats = get_initial_pair_stats(word_freqs)

    num_merges_needed = vocab_size - len(vocab)
    if verbose:
        print(f"Starting BPE merge process for {num_merges_needed} merges...")
    for i in range(num_merges_needed):
        if not pair_stats:
            if verbose:
                print(f"Stopping early after {i} merges: no more pairs to merge.")
            break

        # find best pair from stats
        max_count = max(pair_stats.values())
        candidates = [p for p, c in pair_stats.items() if c == max_count]
        best_pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        # print(f"Merge {i+1}, count={max_count}: {[(vocab[c[0]], vocab[c[1]]) for c in candidates]}")
        # if verbose:
            # print(f"Merge {i+1}, count={max_count}: {candidates}")

        new_token_id = 256 + len(special_tokens) + i

        byte1 = vocab[best_pair[0]]
        byte2 = vocab[best_pair[1]]
        merges.append((byte1, byte2))
        vocab[new_token_id] = byte1 + byte2

        word_freqs = merge_pair_and_update_stats(word_freqs, best_pair, new_token_id, pair_stats)

        if verbose and (i + 1) % 100 == 0:
            print(f"Merge {i+1}/{num_merges_needed} complete.")
    
    if verbose:
        print("BPE training finished.")
    return vocab, merges
