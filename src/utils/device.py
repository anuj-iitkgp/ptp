"""
Hardware and device detection utilities.
Supports CUDA GPUs, Apple Silicon MPS, and CPU fallback.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, Any
import os
import torch


def get_device(device_str: str = "auto") -> torch.device:
    """
    Select the best available compute device or parse explicit user preference.

    Priority:
        CUDA -> Apple Silicon MPS -> CPU
    """
    if device_str is None or device_str.lower() in ("auto", ""):
        env_device = os.getenv("DEVICE", "auto").lower()
        if env_device != "auto":
            device_str = env_device

    if device_str in ("auto", "", None):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    device_str = str(device_str).lower().strip()
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        print("[WARN] CUDA requested but not available. Falling back to auto.")
        return get_device("auto")
    elif device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[WARN] MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def get_dtype(dtype_str: str = "float32", device: torch.device = None) -> torch.dtype:
    """
    Select torch dtype based on string and target device.
    For MPS and CPU, float32 is default for numerical stability unless float16 requested.
    """
    if dtype_str is None:
        dtype_str = os.getenv("DTYPE", "float32")
    dtype_str = str(dtype_str).lower().strip()

    if dtype_str in ("float16", "fp16", "half"):
        # CPU generally does not support fp16 operations well
        if device is not None and device.type == "cpu":
            return torch.float32
        return torch.float16
    elif dtype_str in ("bfloat16", "bf16"):
        return torch.bfloat16
    else:
        return torch.float32


def get_device_info() -> Dict[str, Any]:
    """Return human-readable device diagnostic details."""
    device = get_device("auto")
    info = {
        "device_type": device.type,
        "device_str": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
    elif device.type == "mps":
        info["device_name"] = "Apple Silicon (MPS Accelerator)"
        info["vram_gb"] = "Unified Memory"
    else:
        info["device_name"] = "CPU (Host Processor)"
        info["vram_gb"] = "System RAM"

    return info
