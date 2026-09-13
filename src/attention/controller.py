"""
Core Attention Controller architecture and custom modern Diffusers Attention Processor.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import abc
import math
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention


class AttentionControl(abc.ABC):
    """
    Abstract base class for attention manipulation in diffusion models.
    """

    def __init__(self):
        self.cur_step: int = 0
        self.num_steps: int = 50
        self.cur_att_layer: int = 0
        self.total_layers: int = 0

    def step_callback(self) -> None:
        """Called at the end of each diffusion timestep."""
        self.cur_step += 1
        self.cur_att_layer = 0

    def reset(self) -> None:
        """Reset step and layer counters."""
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(
        self,
        attn_probs: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
    ) -> torch.Tensor:
        """
        Intercept and modify attention probabilities.

        Args:
            attn_probs: Tensor of shape (batch_size * num_heads, query_seq_len, key_seq_len).
            is_cross: True for cross-attention (attn2), False for self-attention (attn1).
            place_in_unet: 'down', 'mid', or 'up'.

        Returns:
            Modified attention probabilities of the same shape.
        """
        raise NotImplementedError


class EmptyControl(AttentionControl):
    """Pass-through controller that leaves attention maps untouched."""

    def forward(
        self,
        attn_probs: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
    ) -> torch.Tensor:
        self.cur_att_layer += 1
        return attn_probs


class P2PAttnProcessor:
    """
    Custom AttentionProcessor for modern Hugging Face Diffusers (>= 0.30.0).
    Intercepts self- and cross-attention scores in UNet2DConditionModel.
    """

    def __init__(
        self,
        controller: AttentionControl,
        place_in_unet: str = "up",
    ):
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if not is_cross:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Intercept attention probabilities through our controller
        if self.controller is not None:
            attention_probs = self.controller.forward(
                attention_probs,
                is_cross=is_cross,
                place_in_unet=self.place_in_unet,
            )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def register_attention_control(
    pipe,
    controller: AttentionControl,
) -> None:
    """
    Hook the controller into all attention modules of the pipeline's UNet.
    """
    attn_processors = {}
    for name, proc in pipe.unet.attn_processors.items():
        if "down_blocks" in name:
            place = "down"
        elif "mid_block" in name:
            place = "mid"
        elif "up_blocks" in name:
            place = "up"
        else:
            place = "up"

        attn_processors[name] = P2PAttnProcessor(controller, place_in_unet=place)

    pipe.unet.set_attn_processor(attn_processors)
