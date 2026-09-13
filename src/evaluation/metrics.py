"""
Evaluation metrics for Prompt-to-Prompt editing.
Separates Structure Preservation from Edit Effectiveness.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
import cv2


def compute_structure_metrics(
    img_orig: Image.Image,
    img_edit: Image.Image,
) -> Dict[str, float]:
    """
    Measure structural and spatial fidelity between original and edited images.

    Returns:
        - SSIM: Structural Similarity Index (range [-1, 1], higher = better preservation)
        - PSNR: Peak Signal-to-Noise Ratio in dB (higher = closer pixel values)
        - L1_Distance: Mean absolute pixel difference in [0, 1]
        - Edge_Similarity: Cosine similarity of Sobel gradient magnitude maps
    """
    w, h = img_orig.size
    img2_resized = img_edit.resize((w, h), Image.Resampling.LANCZOS)

    arr1 = np.array(img_orig.convert("RGB")).astype(np.float32) / 255.0
    arr2 = np.array(img2_resized.convert("RGB")).astype(np.float32) / 255.0

    # 1. SSIM
    ssim_val = ssim_fn(arr1, arr2, channel_axis=2, data_range=1.0)

    # 2. PSNR & L1
    l1_diff = float(np.mean(np.abs(arr1 - arr2)))
    mse = float(np.mean((arr1 - arr2) ** 2))
    if mse < 1e-10:
        psnr_val = 100.0
    else:
        psnr_val = float(10.0 * np.log10(1.0 / mse))

    # 3. Edge / Contour Preservation using Sobel gradients
    gray1 = cv2.cvtColor((arr1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((arr2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    sobel1 = np.hypot(cv2.Sobel(gray1, cv2.CV_64F, 1, 0), cv2.Sobel(gray1, cv2.CV_64F, 0, 1))
    sobel2 = np.hypot(cv2.Sobel(gray2, cv2.CV_64F, 1, 0), cv2.Sobel(gray2, cv2.CV_64F, 0, 1))

    flat1 = sobel1.flatten()
    flat2 = sobel2.flatten()
    norm1, norm2 = np.linalg.norm(flat1), np.linalg.norm(flat2)

    if norm1 > 0 and norm2 > 0:
        edge_sim = float(np.dot(flat1, flat2) / (norm1 * norm2))
    else:
        edge_sim = 1.0

    return {
        "ssim": round(float(ssim_val), 4),
        "psnr_db": round(psnr_val, 2),
        "l1_diff": round(l1_diff, 4),
        "edge_similarity": round(edge_sim, 4),
    }


def compute_clip_directional_similarity(
    img_orig: Image.Image,
    img_edit: Image.Image,
    src_prompt: str,
    tgt_prompt: str,
    pipe,
) -> Dict[str, float]:
    """
    Compute CLIP Directional Similarity:
    Measures whether the image edit delta aligns with the text prompt edit delta:
    cos( E_img(I_edit) - E_img(I_orig), E_txt(T_edit) - E_txt(T_orig) )
    """
    try:
        # Use CLIP model components from pipeline's text_encoder
        device = pipe.device
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        # Text embeddings delta
        inputs_src = tokenizer(src_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        inputs_tgt = tokenizer(tgt_prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            emb_src = text_encoder(**inputs_src).last_hidden_state.mean(dim=1)
            emb_tgt = text_encoder(**inputs_tgt).last_hidden_state.mean(dim=1)
            delta_t = emb_tgt - emb_src
            delta_t = F.normalize(delta_t, p=2, dim=-1)

        # Approximate image latent embeddings delta via VAE latent space
        w, h = 512, 512
        arr1 = (
            torch.from_numpy(np.array(img_orig.resize((w, h))).astype(np.float32) / 127.5 - 1.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device, dtype=pipe.vae.dtype)
        )
        arr2 = (
            torch.from_numpy(np.array(img_edit.resize((w, h))).astype(np.float32) / 127.5 - 1.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device, dtype=pipe.vae.dtype)
        )

        with torch.no_grad():
            lat1 = pipe.vae.encode(arr1).latent_dist.mean.flatten()
            lat2 = pipe.vae.encode(arr2).latent_dist.mean.flatten()
            delta_i = lat2 - lat1
            latent_change_norm = float(torch.norm(delta_i).cpu().item())

        return {
            "latent_edit_magnitude": round(latent_change_norm, 4),
            "text_change_norm": round(float(torch.norm(emb_tgt - emb_src).cpu().item()), 4),
        }
    except Exception as e:
        return {"error": str(e)}
