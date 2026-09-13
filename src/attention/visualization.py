"""
Attention visualization utilities.
Renders spatial heatmaps and alpha-blended overlays on generated images.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.cm as cm

from src.attention.store import AttentionStore


def get_token_attention_map(
    store: AttentionStore,
    token_idx: int,
    batch_idx: int = 0,
    num_heads: int = 8,
    image_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Extract, aggregate, and upsample cross-attention heatmap for a specific token index.

    Args:
        store: AttentionStore instance containing recorded attention maps.
        token_idx: Index of token (0 to 76).
        batch_idx: 0 for original prompt, 1 for edited prompt (in joint diffusion conditional stream).
        num_heads: Number of attention heads (default 8).
        image_size: Target (width, height) to upsample the heatmap.

    Returns:
        2D NumPy float32 array in [0, 1] of shape (height, width).
    """
    # Average attention across recorded layers: (256, 77)
    # Conditional source is batch_idx 2, conditional target is batch_idx 3 in joint 4-item batch
    cond_batch = 2 + batch_idx if batch_idx in (0, 1) else batch_idx
    avg_map = store.get_average_attention(is_cross=True, batch_idx=cond_batch, num_heads=num_heads)

    if token_idx >= avg_map.shape[1]:
        return np.zeros(image_size, dtype=np.float32)

    token_slice = avg_map[:, token_idx]  # (256,)
    res = int(token_slice.shape[0] ** 0.5)  # typically 16
    map_2d = token_slice.view(res, res).float().numpy()

    # Min-max normalization
    min_val, max_val = map_2d.min(), map_2d.max()
    if max_val > min_val:
        map_2d = (map_2d - min_val) / (max_val - min_val)
    else:
        map_2d = np.zeros_like(map_2d)

    # Upsample to image resolution using bicubic interpolation
    upsampled = cv2.resize(map_2d, image_size, interpolation=cv2.INTER_CUBIC)
    return np.clip(upsampled, 0.0, 1.0)


def create_heatmap_image(
    heatmap: np.ndarray,
    colormap_name: str = "turbo",
) -> Image.Image:
    """
    Convert a 2D float [0, 1] heatmap into a colored PIL Image.
    """
    cmap = cm.get_cmap(colormap_name)
    colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) in [0, 1]
    uint8_img = (colored * 255).astype(np.uint8)
    return Image.fromarray(uint8_img)


def create_attention_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap_name: str = "turbo",
) -> Image.Image:
    """
    Overlay attention heatmap on a base PIL Image with alpha blending.
    """
    w, h = image.size
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    colored_heatmap = create_heatmap_image(heatmap, colormap_name=colormap_name)
    base_rgb = image.convert("RGB")

    blended = Image.blend(base_rgb, colored_heatmap, alpha=alpha)
    return blended


def create_token_comparison_grid(
    img_orig: Image.Image,
    img_edit: Image.Image,
    store: AttentionStore,
    src_token_idx: int,
    tgt_token_idx: int,
    src_token_str: str,
    tgt_token_str: str,
) -> Image.Image:
    """
    Create a 2x2 comparison grid:
    [Orig Image + Heatmap]   [Edited Image + Heatmap]
    [Orig Token Heatmap]     [Edited Token Heatmap]
    """
    w, h = img_orig.size
    map_src = get_token_attention_map(store, src_token_idx, batch_idx=0, image_size=(w, h))
    map_tgt = get_token_attention_map(store, tgt_token_idx, batch_idx=1, image_size=(w, h))

    overlay_src = create_attention_overlay(img_orig, map_src)
    overlay_tgt = create_attention_overlay(img_edit, map_tgt)

    heat_src = create_heatmap_image(map_src)
    heat_tgt = create_heatmap_image(map_tgt)

    # Combine into 2x2 grid
    grid = Image.new("RGB", (w * 2 + 30, h * 2 + 30), color=(20, 20, 25))
    grid.paste(overlay_src, (10, 10))
    grid.paste(overlay_tgt, (w + 20, 10))
    grid.paste(heat_src, (10, h + 20))
    grid.paste(heat_tgt, (w + 20, h + 20))

    return grid
