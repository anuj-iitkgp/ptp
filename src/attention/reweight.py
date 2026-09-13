"""
AttentionReweight controller for Attention Re-weighting (Equalizer).
Amplifies or attenuates cross-attention maps for target semantic tokens.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from src.attention.store import AttentionStore


class AttentionReweight(AttentionStore):
    """
    Attention Equalizer controller.
    Applies custom weight multipliers to cross-attention slices for target tokens.
    """

    def __init__(
        self,
        weight_tensor: torch.Tensor,
        self_replace_steps: float = 0.4,
        num_steps: int = 50,
        normalize: bool = True,
        target_resolutions: Tuple[int, ...] = (16, 32),
    ):
        super().__init__(target_resolutions=target_resolutions)
        self.weight_tensor = weight_tensor
        self.self_replace_steps = self_replace_steps
        self.num_steps = num_steps
        self.normalize = normalize

    def forward(
        self,
        attn_probs: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
    ) -> torch.Tensor:
        super().forward(attn_probs, is_cross, place_in_unet)

        total_heads, seq_len_q, seq_len_k = attn_probs.shape
        w = self.weight_tensor.to(device=attn_probs.device, dtype=attn_probs.dtype)

        current_progress = self.cur_step / max(self.num_steps, 1)

        # Handle both joint generation (batch=4) and single generation (batch=2)
        if total_heads % 4 == 0:
            batch_size = 4
            heads_per_item = total_heads // 4
            reshaped = attn_probs.view(4, heads_per_item, seq_len_q, seq_len_k)

            if is_cross:
                # Target conditional stream (index 3)
                target_attn = reshaped[3]  # (heads, seq_len_q, seq_len_k)
                # Multiply by token weights along key dimension
                len_w = min(len(w), seq_len_k)
                scaled = target_attn[:, :, :len_w] * w[:len_w].view(1, 1, -1)

                if self.normalize:
                    denom = scaled.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    scaled = scaled / denom

                reshaped[3, :, :, :len_w] = scaled
            else:
                # Self-attention injection
                if current_progress <= self.self_replace_steps:
                    reshaped[3] = reshaped[2]
                    reshaped[1] = reshaped[0]

            return reshaped.view(total_heads, seq_len_q, seq_len_k)

        elif total_heads % 2 == 0:
            # Single generation with CFG (batch=2: uncond, cond)
            heads_per_item = total_heads // 2
            reshaped = attn_probs.view(2, heads_per_item, seq_len_q, seq_len_k)

            if is_cross:
                target_attn = reshaped[1]  # conditional stream
                len_w = min(len(w), seq_len_k)
                scaled = target_attn[:, :, :len_w] * w[:len_w].view(1, 1, -1)
                if self.normalize:
                    denom = scaled.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    scaled = scaled / denom
                reshaped[1, :, :, :len_w] = scaled

            return reshaped.view(total_heads, seq_len_q, seq_len_k)

        return attn_probs
