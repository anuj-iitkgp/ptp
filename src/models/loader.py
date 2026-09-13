"""
Model loading and hardware management for Stable Diffusion pipelines.
Supports CUDA, Apple Silicon MPS, and CPU fallback with configurable memory optimizations.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Optional, Union, Dict, Any
import warnings
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

from src.utils.device import get_device, get_dtype
from src.utils.config import load_config


def load_model(
    model_id: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    low_memory_mode: bool = False,
    enable_attention_slicing: bool = True,
    enable_vae_slicing: bool = True,
    local_files_only: bool = False,
) -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion pipeline with device placement and memory configurations.

    Args:
        model_id: HuggingFace model identifier or local directory path.
        device: 'auto', 'cuda', 'mps', or 'cpu' (or torch.device).
        dtype: 'float32', 'float16', or torch.dtype.
        low_memory_mode: Enable memory optimizations for lower VRAM environments.
        enable_attention_slicing: Chunk attention computation across heads.
        enable_vae_slicing: Decode VAE latents in chunks.
        local_files_only: Only load from cached snapshot.

    Returns:
        Configured StableDiffusionPipeline.
    """
    config = load_config()

    if model_id is None:
        model_id = config.get("model", {}).get("model_id", "runwayml/stable-diffusion-v1-5")

    if device is None:
        device_pref = config.get("model", {}).get("device", "auto")
        target_device = get_device(device_pref)
    elif isinstance(device, str):
        target_device = get_device(device)
    else:
        target_device = device

    if dtype is None:
        dtype_pref = config.get("model", {}).get("dtype", "float32")
        target_dtype = get_dtype(dtype_pref, target_device)
    elif isinstance(dtype, str):
        target_dtype = get_dtype(dtype, target_device)
    else:
        target_dtype = dtype

    print(f"[Model Loader] Loading '{model_id}' on {target_device} ({target_dtype})...")

    # Suppress verbose warnings during loading
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Attempt first with local_files_only if cached, fallback to online download
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": target_dtype,
            "safety_checker": None,
            "feature_extractor": None,
        }

        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                local_files_only=True,
                **load_kwargs,
            )
            print(f"[Model Loader] Loaded '{model_id}' from local cache successfully.")
        except Exception:
            print(f"[Model Loader] Local cache miss or incomplete snapshot. Fetching '{model_id}'...")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                **load_kwargs,
            )

    # Use DDIM Scheduler for deterministic and invertible steps
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Memory optimizations
    if low_memory_mode or config.get("model", {}).get("low_memory_mode", False):
        enable_attention_slicing = True
        enable_vae_slicing = True

    if enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception as e:
            print(f"[Model Loader] Warning: Could not enable attention slicing: {e}")

    if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception as e:
            print(f"[Model Loader] Warning: Could not enable VAE slicing: {e}")

    # Transfer pipeline to target device
    pipe = pipe.to(target_device)

    print(f"[Model Loader] Model successfully ready on {target_device}!")
    return pipe
