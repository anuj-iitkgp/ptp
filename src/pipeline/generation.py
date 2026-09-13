"""
Text-to-image generation utilities with attention map extraction.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Optional, Tuple, Callable
from PIL import Image
from diffusers import StableDiffusionPipeline

from src.pipeline.p2p_pipeline import PromptToPromptPipeline
from src.attention.store import AttentionStore


def generate_with_attention(
    pipe: StableDiffusionPipeline,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Image.Image, AttentionStore]:
    """
    Generate an image from text while recording cross-attention maps for all prompt tokens.

    Returns:
        (image, attention_store)
    """
    p2p = PromptToPromptPipeline(pipe)
    store = AttentionStore()
    image, store = p2p.generate(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        height=height,
        width=width,
        controller=store,
        callback=callback,
    )
    return image, store
