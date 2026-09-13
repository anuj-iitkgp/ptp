"""
Unit tests for prompt token alignment and parsing.

Author: Anuj Yadav (IIT Kharagpur)
"""

import pytest
from src.prompts.alignment import align_tokens, needleman_wunsch
from src.prompts.parsing import parse_prompt_weights


def test_needleman_wunsch_exact_match():
    seq1 = ["photo", "of", "a", "dog"]
    seq2 = ["photo", "of", "a", "cat"]
    alignment = needleman_wunsch(seq1, seq2)

    assert len(alignment) == 4
    assert alignment[0] == (0, 0)
    assert alignment[1] == (1, 1)
    assert alignment[2] == (2, 2)
    assert alignment[3] == (3, 3)


def test_align_tokens_single_word_swap():
    src = ["<|startoftext|>", "a", "photo", "of", "a", "dog", "on", "beach", "<|endoftext|>"]
    tgt = ["<|startoftext|>", "a", "photo", "of", "a", "cat", "on", "beach", "<|endoftext|>"]

    result = align_tokens(src, tgt, max_length=15)

    assert len(result["changed_tokens"]) == 1
    ch = result["changed_tokens"][0]
    assert ch["src"] == "dog"
    assert ch["tgt"] == "cat"
    assert ch["src_idx"] == 5
    assert ch["tgt_idx"] == 5

    # Target index 5 ("cat") should map to source index 5 ("dog")
    assert result["mapping"][5] == 5


def test_align_tokens_multi_token_expansion():
    # 1 token "dog" replaced by 2 tokens "golden", "retriever"
    src = ["<|startoftext|>", "a", "dog", "running", "<|endoftext|>"]
    tgt = ["<|startoftext|>", "a", "golden", "retriever", "running", "<|endoftext|>"]

    result = align_tokens(src, tgt, max_length=15)

    # Both "golden" and "retriever" should map back to "dog" (src index 2)
    tgt_golden_idx = 2
    tgt_retriever_idx = 3
    assert result["mapping"][tgt_golden_idx] == 2
    assert result["mapping"][tgt_retriever_idx] == 2


def test_align_tokens_prompt_refinement_additions():
    # Add descriptive tokens "small", "brown"
    src = ["<|startoftext|>", "a", "dog", "sitting", "<|endoftext|>"]
    tgt = ["<|startoftext|>", "a", "small", "brown", "dog", "sitting", "<|endoftext|>"]

    result = align_tokens(src, tgt, max_length=15)

    # "a" (tgt 1) -> "a" (src 1)
    assert result["mapping"][1] == 1
    # "dog" (tgt 4) -> "dog" (src 2)
    assert result["mapping"][4] == 2
    # "sitting" (tgt 5) -> "sitting" (src 3)
    assert result["mapping"][5] == 3


def test_parse_prompt_weights_inline_and_dict():
    tokens = ["<|startoftext|>", "a", "red", "car", "on", "street", "<|endoftext|>"]
    prompt = "a red car on street"
    weights_dict = {"car": 2.0, "street": 0.5}

    clean_prompt, weights = parse_prompt_weights(prompt, tokens, weights_dict=weights_dict, max_length=10)

    assert clean_prompt == "a red car on street"
    assert weights[3].item() == pytest.approx(2.0)  # "car"
    assert weights[5].item() == pytest.approx(0.5)  # "street"
    assert weights[1].item() == pytest.approx(1.0)  # "a" unmodified
