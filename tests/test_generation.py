"""
Unit tests for seed reproducibility, device utilities, and image conversion.

Author: Anuj Yadav (IIT Kharagpur)
"""

import pytest
import torch
from PIL import Image
import numpy as np

from src.utils.seed import get_generator, seed_everything
from src.utils.device import get_device, get_dtype
from src.utils.image import create_side_by_side


def test_seed_determinism():
    seed = 1234
    gen1 = get_generator(seed, torch.device("cpu"))
    t1 = torch.randn(4, 4, generator=gen1)

    gen2 = get_generator(seed, torch.device("cpu"))
    t2 = torch.randn(4, 4, generator=gen2)

    assert torch.equal(t1, t2)


def test_seed_differentiation():
    gen1 = get_generator(42, torch.device("cpu"))
    t1 = torch.randn(4, 4, generator=gen1)

    gen2 = get_generator(99, torch.device("cpu"))
    t2 = torch.randn(4, 4, generator=gen2)

    assert not torch.equal(t1, t2)


def test_device_detection():
    dev = get_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")

    dev_cpu = get_device("cpu")
    assert dev_cpu.type == "cpu"


def test_create_side_by_side():
    img1 = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255)

    sbs = create_side_by_side(img1, img2, label1="Left", label2="Right")
    assert isinstance(sbs, Image.Image)
    assert sbs.width > 128
    assert sbs.height > 64
