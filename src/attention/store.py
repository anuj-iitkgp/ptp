"""
Attention map storage and aggregation for visualization.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from src.attention.controller import AttentionControl


class AttentionStore(AttentionControl):
    """
    Controller that records cross-attention and self-attention maps during diffusion.
    """

    def __init__(self, target_resolutions: Tuple[int, ...] = (16, 32)):
        super().__init__()
        self.step_store: Dict[str, List[torch.Tensor]] = {}
        self.attention_store: Dict[str, List[torch.Tensor]] = {}
        self.target_resolutions = target_resolutions

    def step_callback(self) -> None:
        """Store step attentions into aggregate dictionary."""
        if len(self.step_store) > 0:
            for key, val in self.step_store.items():
                if key not in self.attention_store:
                    self.attention_store[key] = []
                self.attention_store[key].append(torch.stack(val, dim=0).mean(dim=0))
            self.step_store = {}
        super().step_callback()

    def reset(self) -> None:
        super().reset()
        self.step_store = {}
        self.attention_store = {}

    def forward(
        self,
        attn_probs: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
    ) -> torch.Tensor:
        """
        Record cross-attention maps for target resolutions.
        attn_probs: shape (batch_size * num_heads, seq_len_q, seq_len_k)
        """
        # Only store cross-attention maps at selected resolutions to optimize RAM
        if is_cross:
            seq_len = attn_probs.shape[1]
            res = int(seq_len ** 0.5)
            if res in self.target_resolutions:
                key = f"{place_in_unet}_{res}_cross"
                if key not in self.step_store:
                    self.step_store[key] = []
                # Detach and move to CPU to conserve device memory
                self.step_store[key].append(attn_probs.detach().cpu())

        self.cur_att_layer += 1
        return attn_probs

    def get_average_attention(
        self,
        is_cross: bool = True,
        batch_idx: int = 0,
        num_heads: int = 8,
    ) -> torch.Tensor:
        """
        Compute averaged attention maps across all recorded layers and timesteps.

        Returns:
            Tensor of shape (seq_len_q, seq_len_k), where seq_len_k=77 for cross-attention.
        """
        key_suffix = "cross" if is_cross else "self"
        matching_keys = [k for k in self.attention_store.keys() if k.endswith(key_suffix)]

        if not matching_keys:
            # Fallback to whatever was recorded in step_store if called before step_callback
            matching_keys = [k for k in self.step_store.keys() if k.endswith(key_suffix)]
            store = self.step_store
        else:
            store = self.attention_store

        if not matching_keys:
            return torch.zeros((256, 77))

        all_maps = []
        for k in matching_keys:
            maps = store[k]  # list of tensors
            for m in maps:
                # Shape: (total_batch_heads, seq_len_q, seq_len_k)
                total_b = m.shape[0]
                # Extract batch_idx heads: [batch_idx * num_heads : (batch_idx + 1) * num_heads]
                start_h = batch_idx * num_heads
                end_h = min(start_h + num_heads, total_b)
                if start_h < total_b:
                    head_slice = m[start_h:end_h].mean(dim=0)  # (seq_len_q, seq_len_k)
                    all_maps.append(head_slice)

        if not all_maps:
            return torch.zeros((256, 77))

        # Standardize query resolution to 16x16 (256) for averaging
        standardized = []
        for m in all_maps:
            seq_len = m.shape[0]
            res = int(seq_len ** 0.5)
            if res != 16:
                # Interpolate to 16x16
                m_2d = m.view(res, res, -1).permute(2, 0, 1).unsqueeze(0)  # (1, 77, res, res)
                m_interp = torch.nn.functional.interpolate(
                    m_2d, size=(16, 16), mode="bilinear", align_corners=False
                ).squeeze(0).permute(1, 2, 0).view(256, -1)
                standardized.append(m_interp)
            else:
                standardized.append(m)

        return torch.stack(standardized, dim=0).mean(dim=0)
