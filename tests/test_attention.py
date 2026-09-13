"""
Unit tests for Attention Controllers and custom Diffusers Processor math.

Author: Anuj Yadav (IIT Kharagpur)
"""

import pytest
import torch

from src.attention.controller import AttentionControl, EmptyControl
from src.attention.store import AttentionStore
from src.attention.replace import AttentionReplace
from src.attention.refine import AttentionRefine
from src.attention.reweight import AttentionReweight


def test_attention_store_recording():
    store = AttentionStore(target_resolutions=(16,))
    store.num_steps = 5

    # Simulate cross-attention batch: (8 heads, 256 pixels, 77 tokens)
    dummy_cross = torch.rand(8, 256, 77)
    out = store.forward(dummy_cross, is_cross=True, place_in_unet="up")
    assert torch.equal(out, dummy_cross)

    store.step_callback()
    avg_map = store.get_average_attention(is_cross=True, batch_idx=0, num_heads=8)
    assert avg_map.shape == (256, 77)


def test_attention_replace_cross_attention_swap():
    # Mapping where target token 5 ("cat") maps to source token 5 ("dog")
    mapping = [i for i in range(77)]
    mapping[5] = 5

    ctrl = AttentionReplace(
        mapping=mapping,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        num_steps=10,
        num_heads=2,
    )
    ctrl.cur_step = 2  # 2 / 10 = 0.2 <= 0.8 (within active window)

    # 4 items: 0=uncond_src, 1=uncond_tgt, 2=cond_src, 3=cond_tgt, each 2 heads -> 8 heads total
    total_heads = 4 * 2
    dummy_cross = torch.zeros(total_heads, 256, 77)
    # Set distinct value for source conditional at token 5
    dummy_cross[4:6, :, 5] = 42.0  # Heads for batch item 2 (cond_src)

    out = ctrl.forward(dummy_cross, is_cross=True, place_in_unet="up")
    reshaped = out.view(4, 2, 256, 77)

    # Verify target conditional (batch item 3) received the source attention map at token 5
    assert torch.allclose(reshaped[3, :, :, 5], torch.tensor(42.0))


def test_attention_replace_self_attention_injection():
    mapping = [i for i in range(77)]
    ctrl = AttentionReplace(
        mapping=mapping,
        cross_replace_steps=0.8,
        self_replace_steps=0.5,
        num_steps=10,
        num_heads=2,
    )
    ctrl.cur_step = 2  # 2 / 10 = 0.2 <= 0.5 (self-attention active)

    total_heads = 4 * 2
    dummy_self = torch.zeros(total_heads, 256, 256)
    # Set unique source self-attention
    dummy_self[4:6] = 99.0  # item 2 (cond_src)

    out = ctrl.forward(dummy_self, is_cross=False, place_in_unet="up")
    reshaped = out.view(4, 2, 256, 256)

    # Target conditional (item 3) should match source conditional (item 2)
    assert torch.allclose(reshaped[3], torch.tensor(99.0))


def test_attention_refine_unmapped_token_preservation():
    # Prompt refinement: target token 2 is newly added (mapped to None)
    mapping = [0, 1, None, 3, 4] + [i for i in range(5, 77)]

    ctrl = AttentionRefine(
        mapping=mapping,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        num_steps=10,
        num_heads=2,
    )
    ctrl.cur_step = 2

    total_heads = 4 * 2
    dummy_cross = torch.zeros(total_heads, 256, 77)
    # Target item 3 has its own attention at token 2
    dummy_cross[6:8, :, 2] = 777.0  # item 3 (cond_tgt)

    out = ctrl.forward(dummy_cross, is_cross=True, place_in_unet="up")
    reshaped = out.view(4, 2, 256, 77)

    # New token 2 should remain intact (not overwritten by source)
    assert torch.allclose(reshaped[3, :, :, 2], torch.tensor(777.0))


def test_attention_reweight_scaling():
    weights = torch.ones(77)
    weights[3] = 2.5  # amplify token 3 by 2.5x

    ctrl = AttentionReweight(
        weight_tensor=weights,
        self_replace_steps=0.4,
        num_steps=10,
        normalize=False,
    )
    ctrl.cur_step = 2

    total_heads = 4 * 2
    dummy_cross = torch.ones(total_heads, 256, 77)

    out = ctrl.forward(dummy_cross, is_cross=True, place_in_unet="up")
    reshaped = out.view(4, 2, 256, 77)

    # Target conditional (item 3) should have token 3 multiplied by 2.5
    assert torch.allclose(reshaped[3, :, :, 3], torch.tensor(2.5))
    # Unmodified tokens should stay 1.0
    assert torch.allclose(reshaped[3, :, :, 1], torch.tensor(1.0))
