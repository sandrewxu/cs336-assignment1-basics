"""
Contains the Tokenizer class for a bpe tokenizer.
"""

import regex as re              # for pretokenization
import multiprocessing          # for multiprocessing
from collections import Counter # more efficient than dict for frequency counting
import mmap                     # work with raw files
import os                       # for filepath handling
from typing import Iterable, Iterator

GPT2_PAT = br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_GPT2_PAT = re.compile(GPT2_PAT)

# -----------------------------------------------------------------------------
# helper functions for training
def get_stats(pretoken_to_bytes: dict[bytes, tuple[list[bytes], int]]) -> Counter[tuple[bytes, bytes]]:
    """
    given pretoken_to_bytes, count frequences of all pairs of bytes
    """
    pair_freq = Counter()
    for byte_list, freq in pretoken_to_bytes.values():
        for pair in zip(byte_list[:-1], byte_list[1:]):
            pair_freq[pair] += freq
    return pair_freq

def merge(pretoken_to_bytes: dict[bytes, tuple[list[bytes], int]], pair_freq: Counter[tuple[bytes, bytes]], best_pair: tuple[bytes, bytes], new_tok: bytes):
    """
    merge best_pair into new_tok in pretoken_to_bytes, pair_freq
    """
    best1, best2 = best_pair
    deltas = Counter()

    for pretoken, (byte_list, freq) in pretoken_to_bytes.items():
        num_bytes = len(byte_list)    
        if (num_bytes < 2):
            continue

        new_byte_list = []
        i = 0
        modified = False

        while i < num_bytes:
            if byte_list[i] == best1 and i < num_bytes - 1 and byte_list[i+1] == best2:
                new_byte_list.append(new_tok)
                i += 2
                modified = True
            else:
                new_byte_list.append(byte_list[i])
                i += 1
        
        if modified:
            for pair in zip(byte_list[:-1], byte_list[1:]):
                deltas[pair] -= freq
            for pair in zip(new_byte_list[:-1], new_byte_list[1:]):
                deltas[pair] += freq
            pretoken_to_bytes[pretoken] = (new_byte_list, freq)
    
    pair_freq.update(deltas)
    pairs_to_remove = [pair for pair, count in pair_freq.items() if count <= 0]
    for pair in pairs_to_remove:
        del pair_freq[pair]
    if best_pair in pair_freq:
       del pair_freq[best_pair]

def process_chunk(input_path: str, start: int, end: int, encoded_special_tokens: list[bytes], split_pattern_bytes: bytes):
    """
    Worker function to read a file chunk, pre-tokenize using pre-compiled pattern, and count pretoken frequencies.
    """
    chunk_pretok_freq = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end-start)
    
    # chunk_text = chunk_bytes.decode("UTF-8", errors="ignore")

    if encoded_special_tokens and split_pattern_bytes:
        segments = re.split(split_pattern_bytes, chunk_bytes)
    else:
        segments = [chunk_bytes]

    for seg in segments:
        if not seg:
            continue
        matches = COMPILED_GPT2_PAT.finditer(seg)
        for match in matches:
            pretok_bytes = match.group(0)
            chunk_pretok_freq[pretok_bytes] += 1
    
    return chunk_pretok_freq

def find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts based on split_special_token.
    Uses memory mapping for efficient searching.
    May return fewer chunks if the boundaries overlap or the token is not found frequently.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    file_size = os.path.getsize(file_path)

    if file_size == 0:
        return [0]
    if desired_num_chunks <= 0:
        return [0, file_size]
    
    safe_desired_chunks = max(1, min(desired_num_chunks, file_size))
    boundaries = {0}
    last_boundary = 0

    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            chunk_size = file_size // safe_desired_chunks

            for i in range(1, safe_desired_chunks):
                ideal_pos = i * chunk_size
                search_start = max(ideal_pos, last_boundary)

                if search_start >= file_size:
                    break

                found_at = mm.find(split_special_token, search_start)

                if found_at != -1:
                    if found_at > last_boundary:
                        boundaries.add(found_at)
                        last_boundary = found_at
                else:
                    break
            boundaries.add(file_size)
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(list(boundaries))

# -----------------------------------------------------------------------------
class Tokenizer:
    """Class for BPE tokenizer"""

    def __init__(self, vocab: dict[int, bytes] | None = None, merges: list[tuple[bytes, bytes]] | None = None, special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens.
        """
        raise NotImplementedError
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        raise NotImplementedError


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields
        token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly
        load into memory.
        """
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        raise NotImplementedError

    
    def train(self, input_path: str, vocab_size: int, special_tokens: list[str], verbose: bool = False) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Given an input_path (str), vocab_size (int, >= 256), and special_tokens (list[str]), construct a vocab (dict[int, bytes]) and a list of merges.
        """
        if self.vocab or self.merges:
            print(f"ERROR: vocab or merges is non-empty.")
            return
        
        # STEP 0: INITIALIZATION
        num_special_tokens = len(special_tokens)
        num_merges = vocab_size - num_special_tokens - 256
        assert(num_merges >= 0)
        vocab = {i: bytes([i]) for i in range(256)}     # vocab number, byte representation
        if verbose:
            print(f"beginning tokenization with {num_merges} merges and {num_special_tokens} special tokens with a vocab size of {vocab_size}")
    
        # STEP 1: PRETOKENIZE THE INPUT
        pretoken_to_bytes = {}     # pretoken (str) mapped to tuple[token_list: list[bytes], freq: int]

        num_processes = multiprocessing.cpu_count()
        if verbose:
            print(f"multiprocessing initialized with {num_processes} cores")

        encoded_special_tokens = None
        split_pattern_bytes = None
        if special_tokens:
            encoded_special_tokens = [st.encode("utf-8") for st in special_tokens]
            split_pattern_bytes = b"|".join(re.escape(st) for st in encoded_special_tokens)

        boundary_token = encoded_special_tokens[0] if encoded_special_tokens else b'\n'
        boundaries = find_chunk_boundaries(input_path, desired_num_chunks=num_processes, split_special_token=boundary_token)
        if verbose:
            print(f"found {len(boundaries) - 1} chunks based on boundary token {boundary_token.decode("utf-8")}")
        
        tasks = [(input_path, start, end, encoded_special_tokens, split_pattern_bytes) for start, end in zip(boundaries[:-1], boundaries[1:])]

        aggregated_pretok_freq = Counter()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(process_chunk, tasks)

            if verbose:
                print(f"aggregating results from {len(results)} processes...")
            for chunk_counter in results:
                aggregated_pretok_freq.update(chunk_counter)

        for pretok_bytes, freq in aggregated_pretok_freq.items():
            # Convert the bytes object into a list of single-byte bytes objects
            byte_list = [bytes([b]) for b in pretok_bytes]
            pretoken_to_bytes[pretok_bytes] = (byte_list, freq) # Store with bytes key
        if verbose:
            print("pretokenization and frequency aggregation complete.")

        # STEP 2: MERGE (given pretokens as strings mapped to list of bytes and pretokens as string mapped to frequency, merge)
        merges = []
        pair_freq = get_stats(pretoken_to_bytes)  # pair of bytes mapped to frequency
        if verbose:
            print("beginning merge")
        for iter in range(num_merges):
            if verbose and not pair_freq:
                print(f"No more pairs to merge. Stopping after {iter} merges")
            
            # find the most frequent pair of bytes, with ties broken by greater lexicographical order
            best_pair = max(pair_freq, key = lambda pair: (pair_freq[pair], pair))

            # update best_pair in vocab, merges
            assert(isinstance(best_pair, tuple) and
                len(best_pair) == 2 and
                isinstance(best_pair[0], bytes) and 
                isinstance(best_pair[1], bytes))
            merges.append((best_pair[0], best_pair[1]))
            vocab[256 + iter] = best_pair[0] + best_pair[1]

            # merge best_pair into vocab[256 + iter] in pretoken_to_bytes, pair_freq
            merge(pretoken_to_bytes, pair_freq, best_pair, vocab[256 + iter])

            if verbose and iter % 100 == 99:
                print(f"{iter + 1}/{num_merges} merges completed")

        # STEP 3: SPECIAL TOKENS
        if verbose:
            print("adding special tokens")
        for i in range(num_special_tokens):
            vocab[256 + num_merges + i] = special_tokens[i].encode("utf-8")
        
        self.vocab = vocab
        self.merges = merges

    def save(self, ):
        raise NotImplementedError

    def load(self, ):
        raise NotImplementedError
    
