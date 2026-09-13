"""
Image processing, latent conversions, base64 encoding, and side-by-side grids.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import List, Union
import io
import base64
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def latent_to_pil(latents: torch.Tensor, vae) -> List[Image.Image]:
    """
    Decode latent tensors z into PIL Images using the VAE decoder.
    Latents shape: (B, 4, H, W)
    """
    # Scale latents with VAE scaling factor (typically 1 / 0.18215 for SD 1.x)
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    scaled_latents = latents / scaling_factor

    with torch.no_grad():
        # Ensure latents match VAE dtype and device
        scaled_latents = scaled_latents.to(dtype=vae.dtype, device=vae.device)
        image = vae.decode(scaled_latents).sample

    # Image tensor: (B, 3, H, W) in range [-1, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    pil_images = []
    for img_arr in image:
        img_uint8 = (img_arr * 255).round().astype(np.uint8)
        pil_images.append(Image.fromarray(img_uint8))

    return pil_images


def pil_to_latent(
    image: Image.Image,
    vae,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    size: int = 512,
) -> torch.Tensor:
    """
    Encode a PIL Image into latent tensor z_0 using the VAE encoder.
    Returns tensor of shape (1, 4, H//8, W//8).
    """
    image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    img_arr = np.array(image).astype(np.float32) / 255.0  # [0, 1]
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    img_tensor = (img_tensor * 2.0 - 1.0).to(device=device, dtype=vae.dtype)

    with torch.no_grad():
        posterior = vae.encode(img_tensor).latent_dist
        latent = posterior.mean  # Use mean for deterministic inversion
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        latent = latent * scaling_factor

    return latent.to(dtype=dtype)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded data URI string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


def create_side_by_side(
    img1: Image.Image,
    img2: Image.Image,
    label1: str = "Original",
    label2: str = "Edited (Prompt-to-Prompt)",
    padding: int = 20,
    header_height: int = 40,
) -> Image.Image:
    """
    Create a labeled side-by-side comparison image of original and edited images.
    """
    w1, h1 = img1.size
    w2, h2 = img2.size
    target_height = max(h1, h2)

    # Resize if heights differ
    if h1 != target_height:
        img1 = img1.resize((int(w1 * target_height / h1), target_height), Image.Resampling.LANCZOS)
    if h2 != target_height:
        img2 = img2.resize((int(w2 * target_height / h2), target_height), Image.Resampling.LANCZOS)

    w1, h1 = img1.size
    w2, h2 = img2.size

    total_width = w1 + w2 + padding * 3
    total_height = target_height + header_height + padding * 2

    # Create dark background canvas
    canvas = Image.new("RGB", (total_width, total_height), color=(26, 28, 35))
    draw = ImageDraw.Draw(canvas)

    # Paste images
    canvas.paste(img1, (padding, header_height + padding))
    canvas.paste(img2, (w1 + padding * 2, header_height + padding))

    # Add text banners
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((padding + 10, padding // 2 + 5), label1, fill=(240, 240, 240), font=font)
    draw.text((w1 + padding * 2 + 10, padding // 2 + 5), label2, fill=(80, 220, 150), font=font)

    return canvas
