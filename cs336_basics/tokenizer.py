"""
Contains the Tokenizer class for a bpe tokenizer.
"""

# -----------------------------------------------------------------------------
# helper functions for training



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

    
    def train(input_path: str, vocab_size: int, special_tokens: list[str], verbose: bool = False, save: str = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        if self.vocab or self.merges:
            print(f"ERROR: vocab or merges is non-empty.")
            return
        
        raise NotImplementedError

    def save():
        raise NotImplementedError

    def load():
        raise NotImplementedError
    
