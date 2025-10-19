from typing import Iterable, Iterator
import json
import regex as re

# GPT-2 pretokenization pattern
GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Returns a mapping from bytes to unicode string for printable display."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))

class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        
        # Build reverse vocab for encoding: bytes -> id
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # Build merge ranks for efficient BPE application
        # Lower rank = earlier merge = higher priority
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, 
        special_tokens: list[str] | None = None):
        # Create GPT-2 byte decoder (reverse of gpt2_bytes_to_unicode)
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        
        # Load vocab from JSON file
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            str_to_id = json.load(f)
        
        # Convert vocab from {token_string: id} to {id: bytes}
        cls_vocab = {}
        for token_str, token_id in str_to_id.items():
            # Convert GPT-2 unicode representation back to bytes
            token_bytes = bytes([byte_decoder[char] for char in token_str])
            cls_vocab[token_id] = token_bytes
        
        # Load merges from text file
        cls_merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line and len(line.split(' ')) == 2:
                    token1_str, token2_str = line.split(' ')
                    # Convert GPT-2 unicode representation back to bytes
                    token1_bytes = bytes([byte_decoder[char] for char in token1_str])
                    token2_bytes = bytes([byte_decoder[char] for char in token2_str])
                    cls_merges.append((token1_bytes, token2_bytes))
        
        return cls(cls_vocab, cls_merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        1. pretokenize the text
        2. apply merges in same order of creation
        """
        if not text:
            return []
        
        token_ids = []
        
        # Split by special tokens if any
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping tokens correctly
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(s) for s in sorted_special_tokens)
            chunks = re.split(f"({special_pattern})", text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            if not chunk:
                continue
            
            # If this chunk is a special token, encode it directly
            if chunk in self.special_tokens:
                special_bytes = chunk.encode('utf-8')
                if special_bytes in self.byte_to_id:
                    token_ids.append(self.byte_to_id[special_bytes])
                continue
            
            # Pretokenize using GPT-2 pattern
            for match in GPT2_PAT.finditer(chunk):
                word = match.group(0)
                word_bytes = word.encode('utf-8')
                
                # Apply BPE to this word
                word_tokens = self._apply_bpe(word_bytes)
                token_ids.extend(word_tokens)
        
        return token_ids
    
    def _apply_bpe(self, word_bytes: bytes) -> list[int]:
        """Apply BPE merges to a word represented as bytes."""
        # Start with individual bytes as tokens
        tokens = [bytes([b]) for b in word_bytes]
        
        if len(tokens) <= 1:
            return [self.byte_to_id[t] for t in tokens]
        
        # Keep merging until no more merges are possible
        while len(tokens) > 1:
            # Find the pair with the lowest merge rank (earliest merge)
            min_rank = float('inf')
            min_pair = None
            min_pos = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair
                        min_pos = i
            
            # If no mergeable pair found, we're done
            if min_pair is None:
                break
            
            # Merge the pair at min_pos
            new_token = min_pair[0] + min_pair[1]
            tokens = tokens[:min_pos] + [new_token] + tokens[min_pos + 2:]
        
        # Convert tokens to IDs
        return [self.byte_to_id[t] for t in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files
        that we cannot directly load into memory
        """
        for chunk in iterable:
            # Encode each chunk and yield token IDs one at a time
            token_ids = self.encode(chunk)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        # Concatenate all token bytes
        all_bytes = b''.join(self.vocab[token_id] for token_id in ids)
        # Decode to UTF-8 string
        return all_bytes.decode('utf-8', errors='replace')
