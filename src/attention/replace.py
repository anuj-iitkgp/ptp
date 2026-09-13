"""
AttentionReplace controller for Word Swap editing in Prompt-to-Prompt.
Replaces cross-attention for mapped tokens and injects self-attention for structural preservation.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from src.attention.store import AttentionStore


class AttentionReplace(AttentionStore):
    """
    Prompt-to-Prompt Word Swap controller.
    Replaces cross-attention for swapped tokens during diffusion and
    optionally shares self-attention to preserve scene geometry.
    """

    def __init__(
        self,
        mapping: List[Optional[int]],
        cross_replace_steps: float = 0.8,
        self_replace_steps: float = 0.4,
        cross_replace_start: float = 0.0,
        num_steps: int = 50,
        num_heads: int = 8,
        target_resolutions: Tuple[int, ...] = (16, 32),
    ):
        super().__init__(target_resolutions=target_resolutions)
        self.mapping = mapping
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.cross_replace_start = cross_replace_start
        self.num_steps = num_steps
        self.num_heads = num_heads

    def forward(
        self,
        attn_probs: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
    ) -> torch.Tensor:
        """
        Intercept and replace attention probabilities between source and target prompts.
        attn_probs shape: (total_heads, seq_len_q, seq_len_k)
        where total_heads = 4 * num_heads when batch has [uncond_src, uncond_tgt, cond_src, cond_tgt].
        """
        # First let store record the maps
        super().forward(attn_probs, is_cross, place_in_unet)

        total_heads, seq_len_q, seq_len_k = attn_probs.shape
        expected_batch = 4

        # If batch is not joint (e.g. 4), return unchanged
        if total_heads % expected_batch != 0:
            return attn_probs

        heads_per_item = total_heads // expected_batch
        # Reshape to (batch=4, heads, seq_q, seq_k)
        # 0: uncond_src, 1: uncond_tgt, 2: cond_src, 3: cond_tgt
        reshaped = attn_probs.view(4, heads_per_item, seq_len_q, seq_len_k)

        current_progress = self.cur_step / max(self.num_steps, 1)

        if is_cross:
            # Cross-Attention Replacement
            if self.cross_replace_start <= current_progress <= self.cross_replace_steps:
                # Target conditional takes Source conditional attention for mapped tokens
                for tgt_idx, src_idx in enumerate(self.mapping):
                    if src_idx is not None and tgt_idx < seq_len_k and src_idx < seq_len_k:
                        # Conditional stream
                        reshaped[3, :, :, tgt_idx] = reshaped[2, :, :, src_idx]
                        # Unconditional stream
                        reshaped[1, :, :, tgt_idx] = reshaped[0, :, :, src_idx]
        else:
            # Self-Attention Injection
            if current_progress <= self.self_replace_steps:
                # Target geometry takes source geometry
                reshaped[3] = reshaped[2]
                reshaped[1] = reshaped[0]

        return reshaped.view(total_heads, seq_len_q, seq_len_k)
