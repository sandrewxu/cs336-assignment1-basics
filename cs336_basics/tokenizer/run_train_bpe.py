import argparse
import base64
import json
import time
import os
from typing import Dict, List, Tuple
from cs336_basics.tokenizer.train_bpe import train_bpe

def gpt2_bytes_to_unicode() -> Dict[int, str]:
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

def save_tokenizer(output_prefix: str, output_path: str, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]):
    """Saves the trained tokenizer vocabulary and merges to disk."""
    # Ensure output_path is a valid directory
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    vocab_file = os.path.join(output_path, f"{output_prefix}.vocab.json")
    merges_file = os.path.join(output_path, f"{output_prefix}.merges.bpe")

    # GPT-2 style byte-to-unicode mapping for readable display
    byte_encoder = gpt2_bytes_to_unicode()
    
    # 1. Save the vocabulary file in GPT-2 format
    # Format: {token_string: id} where bytes are converted using GPT-2 mapping
    str_to_id_vocab = {}
    for idx, token_bytes in vocab.items():
        # Convert bytes to GPT-2's unicode representation
        token_str = ''.join(byte_encoder[b] for b in token_bytes)
        str_to_id_vocab[token_str] = idx
    
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(str_to_id_vocab, f, indent=4, ensure_ascii=False)
    print(f"Vocabulary saved to: {vocab_file}")

    # 2. Save the merges file in GPT-2 format
    with open(merges_file, 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            # Convert bytes to GPT-2 unicode representation
            t1 = ''.join(byte_encoder[b] for b in p1)
            t2 = ''.join(byte_encoder[b] for b in p2)
            f.write(f"{t1} {t2}\n")
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
