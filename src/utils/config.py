"""
Configuration loader with YAML and environment variable support.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, Any
import os
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration dictionary from YAML file and override with environment variables.
    """
    config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Ensure sections exist
    config.setdefault("model", {})
    config.setdefault("diffusion", {})
    config.setdefault("editing", {})
    config.setdefault("paths", {})

    # Apply environment variable overrides
    if "MODEL_ID" in os.environ:
        config["model"]["model_id"] = os.environ["MODEL_ID"]
    if "DEVICE" in os.environ:
        config["model"]["device"] = os.environ["DEVICE"]
    if "DTYPE" in os.environ:
        config["model"]["dtype"] = os.environ["DTYPE"]
    if "LOW_MEMORY_MODE" in os.environ:
        config["model"]["low_memory_mode"] = os.environ["LOW_MEMORY_MODE"].lower() in ("true", "1", "yes")

    if "IMAGE_SIZE" in os.environ:
        config["diffusion"]["image_size"] = int(os.environ["IMAGE_SIZE"])
    if "NUM_INFERENCE_STEPS" in os.environ:
        config["diffusion"]["num_inference_steps"] = int(os.environ["NUM_INFERENCE_STEPS"])
    if "GUIDANCE_SCALE" in os.environ:
        config["diffusion"]["guidance_scale"] = float(os.environ["GUIDANCE_SCALE"])

    if "CROSS_REPLACE_STEPS" in os.environ:
        config["editing"]["cross_replace_steps"] = float(os.environ["CROSS_REPLACE_STEPS"])
    if "SELF_REPLACE_STEPS" in os.environ:
        config["editing"]["self_replace_steps"] = float(os.environ["SELF_REPLACE_STEPS"])

    if "OUTPUT_DIR" in os.environ:
        config["paths"]["output_dir"] = os.environ["OUTPUT_DIR"]

    return config
