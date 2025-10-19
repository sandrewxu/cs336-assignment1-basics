#!/usr/bin/env python3
"""Decode a Base64-encoded BPE vocab file to human-readable format."""
import json
import base64
import sys

def decode_vocab(vocab_path: str, limit: int = 50):
    """Decode and print the first `limit` tokens from a Base64-encoded vocab."""
    with open(vocab_path, 'r') as f:
        encoded_vocab = json.load(f)
    
    print(f"Vocabulary from: {vocab_path}")
    print(f"Total tokens: {len(encoded_vocab)}\n")
    print(f"{'ID':<8} {'Base64':<12} {'Decoded (repr)':<40} {'UTF-8 (if valid)':<30}")
    print("-" * 100)
    
    for idx_str, b64_val in list(encoded_vocab.items())[:limit]:
        idx = int(idx_str)
        # Decode from Base64
        token_bytes = base64.b64decode(b64_val)
        
        # Show repr (shows \x00, \n, etc.)
        repr_str = repr(token_bytes)[2:-1]  # Remove b'...'
        
        # Try to decode as UTF-8 (many merged tokens will be valid UTF-8)
        try:
            utf8_str = token_bytes.decode('utf-8')
            # Escape newlines/tabs for display
            utf8_display = utf8_str.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        except UnicodeDecodeError:
            utf8_display = "(invalid UTF-8)"
        
        print(f"{idx:<8} {b64_val:<12} {repr_str:<40} {utf8_display:<30}")

if __name__ == "__main__":
    vocab_file = sys.argv[1] if len(sys.argv) > 1 else "results/TinyStories/tinystories_train.vocab.json"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    decode_vocab(vocab_file, limit)

