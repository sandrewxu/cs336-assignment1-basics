"""
Decode language model output
"""

from jaxtyping import Int
import torch
from typing import Optional

def generate_output(
    prompt: str,
    generate_until_complete: Optional[bool] = True,
    max_tokens: Optional[int] = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    """
    Decoding function
    """
    # Tokenize prompt
    tokenized_prompt = None
    
    # Untokenize generation


def generate_next_token(
    input: Int[torch.Tensor, "seq_len"],
    
) -> int:
    """
    """
    pass
