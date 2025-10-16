import os
import regex as re
from collections import defaultdict
import multiprocessing
import mmap
import heapq
from array import array
from typing import List, Tuple, Dict, DefaultDict, Set

GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

# --- INTERNAL DATA STRUCTURES (OpenWebText-scale) ---
class WordStore:
    """Stores unique words as token id arrays and their frequencies.

    Also tracks per-word mapping from pair -> positions for fast local updates.
    """

    def __init__(self) -> None:
        self._tokens: Dict[int, array] = {}
        self._freq: Dict[int, int] = {}
        self._pair_positions: Dict[int, Dict[Tuple[int, int], List[int]]] = {}
        self._next_id: int = 0

    def add_word(self, word_tokens: Tuple[int, ...], freq: int) -> int:
        wid = self._next_id
        self._next_id += 1
        # Use 32-bit arrays for simplicity and headroom
        token_arr = array('I', word_tokens)
        self._tokens[wid] = token_arr
        self._freq[wid] = freq
        self._pair_positions[wid] = self._compute_pair_positions(token_arr)
        return wid

    def items(self):
        for wid in self._tokens.keys():
            yield wid, self._tokens[wid], self._freq[wid]

    def get_tokens(self, wid: int) -> array:
        return self._tokens[wid]

    def get_freq(self, wid: int) -> int:
        return self._freq[wid]

    def get_pair_positions(self, wid: int) -> Dict[Tuple[int, int], List[int]]:
        return self._pair_positions[wid]

    def update_tokens_and_pairs(self, wid: int, new_tokens: array) -> None:
        self._tokens[wid] = new_tokens
        self._pair_positions[wid] = self._compute_pair_positions(new_tokens)

    @staticmethod
    def _compute_pair_positions(tokens: array) -> Dict[Tuple[int, int], List[int]]:
        positions: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        if len(tokens) < 2:
            return positions
        # enumerate adjacent pairs and record their starting positions
        prev = tokens[0]
        for i in range(1, len(tokens)):
            cur = tokens[i]
            positions[(prev, cur)].append(i - 1)
            prev = cur
        return positions


class PairIndex:
    """Tracks global pair counts and word membership; provides a lazy max-heap.

    - pair_counts[(a,b)] -> total count across corpus (weighted by word freq)
    - pair_to_word_ids[(a,b)] -> set of word ids that contain the pair
    - heap entries: (-count, tie_key, (a,b)), with lazy invalidation on pop
    """

    def __init__(self) -> None:
        self.pair_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
        self.pair_to_word_ids: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self._heap: List[Tuple[int, Tuple[bytes, bytes], Tuple[int, int]]] = []

    def add_word_pairs(self, wid: int, pair_positions: Dict[Tuple[int, int], List[int]], weight: int) -> None:
        for pair, positions in pair_positions.items():
            if not positions:
                continue
            self.pair_counts[pair] += weight * len(positions)
            self.pair_to_word_ids[pair].add(wid)

    def remove_word_pairs(self, wid: int, pair_positions: Dict[Tuple[int, int], List[int]], weight: int) -> None:
        for pair, positions in pair_positions.items():
            if not positions:
                continue
            self.pair_counts[pair] -= weight * len(positions)
            # pair may still exist in this word after update; set maintenance happens in reconcile_word_membership

    def reconcile_word_membership(self, wid: int, before: Dict[Tuple[int, int], List[int]], after: Dict[Tuple[int, int], List[int]]) -> Set[Tuple[int, int]]:
        """Update pair_to_word_ids sets based on before/after presence.

        Returns the set of pairs whose global count potentially changed (union of keys).
        """
        changed_pairs: Set[Tuple[int, int]] = set(before.keys()) | set(after.keys())
        for pair in changed_pairs:
            had = bool(before.get(pair))
            has = bool(after.get(pair))
            if had and not has:
                ws = self.pair_to_word_ids.get(pair)
                if ws is not None and wid in ws:
                    ws.discard(wid)
                    if not ws:
                        # cleanup to keep structure small
                        self.pair_to_word_ids.pop(pair, None)
            elif has:
                self.pair_to_word_ids[pair].add(wid)
        return changed_pairs

    def push_heap(self, pair: Tuple[int, int], vocab: Dict[int, bytes]) -> None:
        count = self.pair_counts.get(pair, 0)
        if count <= 0:
            return
        a, b = pair
        a_bytes = vocab[a]
        b_bytes = vocab[b]
        # For lexicographically greatest pair first in a min-heap:
        # Negate each byte value and append a sentinel (1) to handle prefix cases
        # This ensures longer sequences with more negative values come first
        tie_key = (
            tuple(-x for x in a_bytes) + (1,),
            tuple(-x for x in b_bytes) + (1,)
        )
        heapq.heappush(self._heap, (-count, tie_key, pair))

    def build_heap(self, vocab: Dict[int, bytes]) -> None:
        self._heap.clear()
        for pair, count in self.pair_counts.items():
            if count > 0:
                self.push_heap(pair, vocab)

    def pop_best_pair(self) -> Tuple[int, int, int]:
        """Pop the pair with highest count; returns (a, b, count). May return (0,0,0) if empty."""
        while self._heap:
            neg_count, _tie, pair = heapq.heappop(self._heap)
            count = -neg_count
            actual = self.pair_counts.get(pair, 0)
            if actual == count and count > 0:
                return pair[0], pair[1], count
            # else stale; continue
        return 0, 0, 0

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
    Only updates pair statistics for pairs affected by the merge.
    """
    new_word_freqs = defaultdict(int)
    p1, p2 = pair_to_merge

    for word, freq in word_freqs.items():
        # only one token, cannot be merged
        if len(word) < 2:
            new_word_freqs[word] += freq
            continue
        
        # Find all positions where the merge occurs and build new word
        new_word = []
        merge_positions = []  # Track positions in new_word where merges happened
        i = 0
        j = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i+1]) == (p1, p2):
                merge_positions.append(j)
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
            j += 1

        if merge_positions:
            # Only update pair stats around merge positions
            # Track which pairs we've already modified to avoid double-counting
            # when consecutive merges affect the same pairs
            old_word_pos = 0
            new_word_pos = 0

            for i, merge_idx in enumerate(merge_positions):
                # Advance to the merge position in new_word
                while new_word_pos < merge_idx:
                    old_word_pos += 1
                    new_word_pos += 1

                is_next_consecutive = (i + 1 < len(merge_positions) and 
                                      merge_positions[i + 1] == merge_idx + 1)
                if old_word_pos > 0:
                    pair_stats[(word[old_word_pos - 1], word[old_word_pos])] -= freq
                if old_word_pos + 2 < len(word):
                    if not is_next_consecutive:
                        pair_stats[(word[old_word_pos + 1], word[old_word_pos + 2])] -= freq
                pair_stats[(word[old_word_pos], word[old_word_pos + 1])] -= freq
                
                if new_word_pos > 0:
                    pair_stats[(new_word[new_word_pos - 1], new_word[new_word_pos])] += freq
                if new_word_pos + 1 < len(new_word):
                    if not is_next_consecutive:
                        pair_stats[(new_word[new_word_pos], new_word[new_word_pos + 1])] += freq
                old_word_pos += 2
                new_word_pos += 1

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
        for res_dict in pool.imap_unordered(process_chunk_mmap, pool_args, chunksize=1):
            for word, freq in res_dict.items():
                word_freqs[word] += freq

    if verbose:
        print(f"Pre-tokenization complete. Found {len(word_freqs)} unique words.")

    # 2. Build WordStore and PairIndex, then perform exact merges with heap
    merges: List[Tuple[bytes, bytes]] = []
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        vocab[256 + i] = token_str.encode('utf-8')

    # initialize WordStore
    ws = WordStore()
    for word_tokens, freq in word_freqs.items():
        ws.add_word(word_tokens, freq)

    # initialize PairIndex with counts and membership
    pi = PairIndex()
    for wid, _tokens, freq in ws.items():
        pi.add_word_pairs(wid, ws.get_pair_positions(wid), freq)

    # build initial heap
    pi.build_heap(vocab)

    def _merge_tokens(tokens: array, p1: int, p2: int, new_id: int) -> array:
        if len(tokens) < 2:
            return array('I', tokens)
        out = array('I')
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == p1 and tokens[i + 1] == p2:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    num_merges_needed = vocab_size - len(vocab)
    if verbose:
        print(f"Starting BPE merge process for {num_merges_needed} merges...")
    for i in range(num_merges_needed):
        a, b, count = pi.pop_best_pair()
        if count == 0:
            if verbose:
                print(f"Stopping early after {i} merges: no more pairs to merge.")
            break

        new_token_id = 256 + len(special_tokens) + i
        byte1 = vocab[a]
        byte2 = vocab[b]
        merges.append((byte1, byte2))
        vocab[new_token_id] = byte1 + byte2

        affected = list(pi.pair_to_word_ids.get((a, b), set()))
        for wid in affected:
            freq = ws.get_freq(wid)
            before_positions = ws.get_pair_positions(wid)
            pi.remove_word_pairs(wid, before_positions, freq)

            tokens_arr = ws.get_tokens(wid)
            new_tokens = _merge_tokens(tokens_arr, a, b, new_token_id)
            ws.update_tokens_and_pairs(wid, new_tokens)

            after_positions = ws.get_pair_positions(wid)
            pi.add_word_pairs(wid, after_positions, freq)
            changed_pairs = pi.reconcile_word_membership(wid, before_positions, after_positions)
            for pair in changed_pairs:
                pi.push_heap(pair, vocab)

        if verbose and (i + 1) % 100 == 0:
            print(f"Merge {i+1}/{num_merges_needed} complete.")
    
    if verbose:
        print("BPE training finished.")
    return vocab, merges
