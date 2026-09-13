"""
Token alignment engine using dynamic programming sequence alignment.
Supports 1-to-1, multi-token substitutions, insertions, and deletions.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import List, Dict, Any, Optional, Tuple


def needleman_wunsch(
    seq1: List[str],
    seq2: List[str],
    match_score: float = 2.0,
    mismatch_score: float = -1.0,
    gap_penalty: float = -1.0,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Perform Needleman-Wunsch global alignment between two token sequences.
    Returns list of aligned pairs (idx1, idx2), where None indicates a gap.
    """
    n, m = len(seq1), len(seq2)
    score_matrix = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        score_matrix[i][0] = i * gap_penalty
    for j in range(m + 1):
        score_matrix[0][j] = j * gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s1 = seq1[i - 1].lower().strip()
            s2 = seq2[j - 1].lower().strip()

            if s1 == s2:
                sim = match_score
            else:
                sim = mismatch_score

            match = score_matrix[i - 1][j - 1] + sim
            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

    # Backtrace
    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s1 = seq1[i - 1].lower().strip()
            s2 = seq2[j - 1].lower().strip()
            sim = match_score if s1 == s2 else mismatch_score

            if score_matrix[i][j] == score_matrix[i - 1][j - 1] + sim:
                aligned.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue

        if i > 0 and (j == 0 or score_matrix[i][j] == score_matrix[i - 1][j] + gap_penalty):
            aligned.append((i - 1, None))
            i -= 1
        else:
            aligned.append((None, j - 1))
            j -= 1

    aligned.reverse()
    return aligned


def align_tokens(
    source_tokens: List[str],
    target_tokens: List[str],
    max_length: int = 77,
) -> Dict[str, Any]:
    """
    Align source and target tokens.
    Returns structured token alignment and mapping indices.

    Format returned:
    {
        "source_tokens": [...],
        "target_tokens": [...],
        "mapping": [src_idx_for_tgt_0, src_idx_for_tgt_1, ...], # Length max_length
        "changed_tokens": [{"src_idx": int, "tgt_idx": int, "src": str, "tgt": str}, ...],
        "unchanged_tokens": [{"src_idx": int, "tgt_idx": int, "token": str}, ...],
        "added_tokens": [{"tgt_idx": int, "tgt": str}, ...],
        "removed_tokens": [{"src_idx": int, "src": str}, ...]
    }
    """
    # Clean token strings for comparison
    clean_src = [t.strip().replace("</w>", "") for t in source_tokens]
    clean_tgt = [t.strip().replace("</w>", "") for t in target_tokens]

    # Align real token spans (up to max_length)
    len_s = min(len(clean_src), max_length)
    len_t = min(len(clean_tgt), max_length)

    alignment_pairs = needleman_wunsch(clean_src[:len_s], clean_tgt[:len_t])

    # Target index to source index mapping array
    mapping: List[Optional[int]] = [None] * max_length
    changed_tokens = []
    unchanged_tokens = []
    added_tokens = []
    removed_tokens = []

    # Track alignment pairs
    last_src_idx = None
    for src_idx, tgt_idx in alignment_pairs:
        if src_idx is not None and tgt_idx is not None:
            mapping[tgt_idx] = src_idx
            s_tok = clean_src[src_idx]
            t_tok = clean_tgt[tgt_idx]
            if s_tok.lower() == t_tok.lower():
                unchanged_tokens.append({"src_idx": src_idx, "tgt_idx": tgt_idx, "token": s_tok})
            else:
                changed_tokens.append({"src_idx": src_idx, "tgt_idx": tgt_idx, "src": s_tok, "tgt": t_tok})
            last_src_idx = src_idx
        elif src_idx is None and tgt_idx is not None:
            # Added token in target (e.g. Prompt refinement or multi-token replacement)
            # If adjacent to a changed token in multi-token replacement, optionally map to last changed src
            t_tok = clean_tgt[tgt_idx]
            added_tokens.append({"tgt_idx": tgt_idx, "tgt": t_tok})
            mapping[tgt_idx] = None
        elif src_idx is not None and tgt_idx is None:
            # Removed token from source
            s_tok = clean_src[src_idx]
            removed_tokens.append({"src_idx": src_idx, "src": s_tok})

    # For multi-token phrases where 1 source word expands to multiple target tokens:
    # e.g., "dog" -> "golden", "retriever"
    # If a changed token is followed by added tokens before the next unchanged token,
    # map those target tokens to the swapped source token so they inherit spatial attention.
    for i in range(len(changed_tokens)):
        ch = changed_tokens[i]
        src_i = ch["src_idx"]
        tgt_i = ch["tgt_idx"]
        # Check if following target tokens are added
        curr_t = tgt_i + 1
        while curr_t < len_t and mapping[curr_t] is None:
            mapping[curr_t] = src_i
            curr_t += 1

    # Fill remaining indices (padding tokens) 1-to-1
    for k in range(max_length):
        if mapping[k] is None and k < len_s:
            mapping[k] = k

    return {
        "source_tokens": clean_src,
        "target_tokens": clean_tgt,
        "mapping": mapping,
        "changed_tokens": changed_tokens,
        "unchanged_tokens": unchanged_tokens,
        "added_tokens": added_tokens,
        "removed_tokens": removed_tokens,
    }
