"""
High-level editing APIs for Prompt-to-Prompt methods:
- Word Swap
- Prompt Refinement
- Attention Re-weighting

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from src.pipeline.p2p_pipeline import PromptToPromptPipeline
from src.prompts.tokenizer import PromptTokenizer
from src.prompts.alignment import align_tokens
from src.prompts.parsing import parse_prompt_weights
from src.attention.replace import AttentionReplace
from src.attention.refine import AttentionRefine
from src.attention.reweight import AttentionReweight
from src.attention.controller import AttentionControl


def edit_word_swap(
    pipe: StableDiffusionPipeline,
    source_prompt: str,
    target_prompt: str,
    cross_replace_steps: float = 0.8,
    self_replace_steps: float = 0.4,
    cross_replace_start: float = 0.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    initial_latents: Optional[torch.Tensor] = None,
    callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Image.Image, Image.Image, AttentionControl, Dict]:
    """
    Perform Prompt-to-Prompt Word Swap.
    Swaps one or more words while preserving scene background and object geometry.

    Returns:
        (original_image, edited_image, controller, alignment_info)
    """
    tokenizer = PromptTokenizer(pipe.tokenizer, pipe.text_encoder)
    src_meta = tokenizer.tokenize(source_prompt)
    tgt_meta = tokenizer.tokenize(target_prompt)

    alignment = align_tokens(src_meta["tokens"], tgt_meta["tokens"])

    controller = AttentionReplace(
        mapping=alignment["mapping"],
        cross_replace_steps=cross_replace_steps,
        self_replace_steps=self_replace_steps,
        cross_replace_start=cross_replace_start,
        num_steps=num_inference_steps,
    )

    p2p = PromptToPromptPipeline(pipe)
    img_orig, img_edit, controller = p2p.edit(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        controller=controller,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        height=height,
        width=width,
        initial_latents=initial_latents,
        callback=callback,
    )

    return img_orig, img_edit, controller, alignment


def edit_prompt_refinement(
    pipe: StableDiffusionPipeline,
    source_prompt: str,
    target_prompt: str,
    cross_replace_steps: float = 0.8,
    self_replace_steps: float = 0.4,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    initial_latents: Optional[torch.Tensor] = None,
    callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Image.Image, Image.Image, AttentionControl, Dict]:
    """
    Perform Prompt-to-Prompt Prompt Refinement.
    Allows new descriptive words to be added while anchoring shared background and composition.

    Returns:
        (original_image, edited_image, controller, alignment_info)
    """
    tokenizer = PromptTokenizer(pipe.tokenizer, pipe.text_encoder)
    src_meta = tokenizer.tokenize(source_prompt)
    tgt_meta = tokenizer.tokenize(target_prompt)

    alignment = align_tokens(src_meta["tokens"], tgt_meta["tokens"])

    controller = AttentionRefine(
        mapping=alignment["mapping"],
        cross_replace_steps=cross_replace_steps,
        self_replace_steps=self_replace_steps,
        num_steps=num_inference_steps,
    )

    p2p = PromptToPromptPipeline(pipe)
    img_orig, img_edit, controller = p2p.edit(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        controller=controller,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        height=height,
        width=width,
        initial_latents=initial_latents,
        callback=callback,
    )

    return img_orig, img_edit, controller, alignment


def edit_reweight(
    pipe: StableDiffusionPipeline,
    prompt: str,
    weights_dict: Dict[str, float],
    self_replace_steps: float = 0.4,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    initial_latents: Optional[torch.Tensor] = None,
    callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Image.Image, Image.Image, AttentionControl, Dict]:
    """
    Perform Attention Re-weighting (Equalizer).
    Scales the attention maps for target tokens to accentuate or dampen specific concepts.

    Returns:
        (original_image, edited_image, controller, alignment_info)
    """
    tokenizer = PromptTokenizer(pipe.tokenizer, pipe.text_encoder)
    meta = tokenizer.tokenize(prompt)
    clean_prompt, weight_tensor = parse_prompt_weights(
        prompt=prompt,
        tokenizer_tokens=meta["tokens"],
        weights_dict=weights_dict,
    )

    controller = AttentionReweight(
        weight_tensor=weight_tensor,
        self_replace_steps=self_replace_steps,
        num_steps=num_inference_steps,
    )

    p2p = PromptToPromptPipeline(pipe)
    # Joint run with same prompt: original (unmodified) vs reweighted
    img_orig, img_edit, controller = p2p.edit(
        source_prompt=clean_prompt,
        target_prompt=clean_prompt,
        controller=controller,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        height=height,
        width=width,
        initial_latents=initial_latents,
        callback=callback,
    )

    alignment = {
        "source_tokens": meta["clean_tokens"],
        "target_tokens": meta["clean_tokens"],
        "reweighted_tokens": weights_dict,
    }

    return img_orig, img_edit, controller, alignment
