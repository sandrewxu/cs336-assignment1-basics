import argparse
import base64
import json
import time
import os
from typing import Dict, List, Tuple
from cs336_basics.tokenizer.train_bpe import train_bpe

def save_tokenizer(output_prefix: str, output_path: str, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]):
    """Saves the trained tokenizer vocabulary and merges to disk."""
    # Ensure output_path is a valid directory
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    vocab_file = os.path.join(output_path, f"{output_prefix}.vocab.json")
    merges_file = os.path.join(output_path, f"{output_prefix}.merges.bpe")

    # 1. Save the vocabulary file
    # We use Base64 to safely encode the bytes for JSON serialization
    encoded_vocab = {
        idx: base64.b64encode(token_bytes).decode('utf-8') 
        for idx, token_bytes in vocab.items()
    }
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(encoded_vocab, f, indent=2, ensure_ascii=False)
    print(f"Vocabulary saved to: {vocab_file}")

    # 2. Save the merges file
    # This format is compatible with many existing BPE implementations
    with open(merges_file, 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            # Writing the raw bytes directly might cause encoding issues,
            # so we represent them in a readable way. Here, we use utf-8 representation.
            # A more robust way might use base64 again, but this is standard.
            f.write(f"{p1.decode('utf-8', 'ignore')} {p2.decode('utf-8', 'ignore')}\n")
    print(f"Merges saved to: {merges_file}")

def main():
    """Main function to parse arguments and run BPE training."""
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on a large text file."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the training data file")
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="The desired final vocabulary size."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Prefix for the output files (e.g. tinystories_tokenizer)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Folder for output files (e.g. results/TinyStories)."
    )
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    start_time = time.time()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        verbose=True
    )

    end_time = time.time()
    print(f"Training completed in {(end_time - start_time):.2f} seconds.")

    save_tokenizer(args.output_prefix, args.output_path, vocab, merges)

if __name__ == "__main__":
    main()
