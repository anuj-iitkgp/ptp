"""
AttentionRefine controller for Prompt Refinement editing in Prompt-to-Prompt.
Preserves attention for shared tokens while allowing newly introduced descriptive tokens to attend freely.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import List, Optional, Tuple
import torch
from src.attention.store import AttentionStore


class AttentionRefine(AttentionStore):
    """
    Prompt Refinement controller.
    Allows new tokens (e.g. adjectives or specifications) to inject fresh semantic attention
    while preserving source cross-attention for all shared contextual tokens.
    """

    def __init__(
        self,
        mapping: List[Optional[int]],
        cross_replace_steps: float = 0.8,
        self_replace_steps: float = 0.4,
        num_steps: int = 50,
        num_heads: int = 8,
        target_resolutions: Tuple[int, ...] = (16, 32),
    ):
        super().__init__(target_resolutions=target_resolutions)
        self.mapping = mapping
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.num_steps = num_steps
        self.num_heads = num_heads

    def forward(
        self,
        attn_probs: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
    ) -> torch.Tensor:
        # Record attention maps
        super().forward(attn_probs, is_cross, place_in_unet)

        total_heads, seq_len_q, seq_len_k = attn_probs.shape
        expected_batch = 4

        if total_heads % expected_batch != 0:
            return attn_probs

        heads_per_item = total_heads // expected_batch
        reshaped = attn_probs.view(4, heads_per_item, seq_len_q, seq_len_k)
        current_progress = self.cur_step / max(self.num_steps, 1)

        if is_cross:
            if current_progress <= self.cross_replace_steps:
                # Only replace attention for tokens that have an alignment in source prompt
                for tgt_idx, src_idx in enumerate(self.mapping):
                    if src_idx is not None and tgt_idx < seq_len_k and src_idx < seq_len_k:
                        # Conditional stream
                        reshaped[3, :, :, tgt_idx] = reshaped[2, :, :, src_idx]
                        # Unconditional stream
                        reshaped[1, :, :, tgt_idx] = reshaped[0, :, :, src_idx]
                    # Note: When src_idx is None, reshaped[3, :, :, tgt_idx] is kept untouched,
                    # allowing the new descriptive token to freely influence the latent representation.
        else:
            # Self-attention structural preservation
            if current_progress <= self.self_replace_steps:
                reshaped[3] = reshaped[2]
                reshaped[1] = reshaped[0]

        return reshaped.view(total_heads, seq_len_q, seq_len_k)
