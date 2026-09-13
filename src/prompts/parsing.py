"""
Prompt parsing utilities for attention reweighting and equalizer specification.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, List, Tuple, Union
import re
import torch


def parse_prompt_weights(
    prompt: str,
    tokenizer_tokens: List[str],
    weights_dict: Union[Dict[str, float], None] = None,
    max_length: int = 77,
) -> Tuple[str, torch.Tensor]:
    """
    Parse equalizer weights from a prompt with '(word:weight)' syntax or from a dictionary.

    Args:
        prompt: Raw prompt text, optionally with syntax 'photo of a (dog:1.8) on a beach'.
        tokenizer_tokens: Decoded token list from the tokenizer.
        weights_dict: Explicit dictionary mapping word/subword -> float multiplier.
        max_length: Maximum sequence length (77 for CLIP).

    Returns:
        clean_prompt: Prompt with weight markers stripped.
        weight_tensor: 1D torch.Tensor of shape (max_length,) with multipliers (1.0 default).
    """
    weights = torch.ones(max_length, dtype=torch.float32)
    explicit_weights: Dict[str, float] = {}

    if weights_dict is not None:
        for k, v in weights_dict.items():
            explicit_weights[k.lower().strip()] = float(v)

    # Parse inline (word:weight) syntax: e.g. (dog:1.5)
    pattern = re.compile(r"\(([^:]+):([0-9\.]+)\)")
    matches = pattern.findall(prompt)
    for word, weight in matches:
        explicit_weights[word.lower().strip()] = float(weight)

    clean_prompt = pattern.sub(r"\1", prompt)

    # Map parsed words to tokenizer tokens
    clean_tokens = [t.lower().strip().replace("</w>", "") for t in tokenizer_tokens]

    for word, weight in explicit_weights.items():
        # Match single or subword tokens
        found = False
        for idx, tok in enumerate(clean_tokens[:max_length]):
            if tok == word:
                weights[idx] = weight
                found = True
            elif len(tok) >= 3 and len(word) >= 3 and (tok.startswith(word) or word.startswith(tok)):
                weights[idx] = weight
                found = True
        if not found:
            print(f"[Parsing] Warning: Word '{word}' for reweighting not found in tokens.")

    return clean_prompt, weights
