"""
Prompt-to-Prompt Master Pipeline.
Executes joint diffusion with attention interception for genuine cross-attention prompt editing.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from src.prompts.tokenizer import PromptTokenizer
from src.prompts.alignment import align_tokens
from src.attention.controller import AttentionControl, EmptyControl, register_attention_control
from src.utils.seed import get_generator, seed_everything
from src.utils.image import latent_to_pil


class PromptToPromptPipeline:
    """
    Orchestrates joint diffusion for original and edited prompts using cross-attention control.
    """

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.tokenizer = PromptTokenizer(pipe.tokenizer, pipe.text_encoder)
        self.device = pipe.device
        self.dtype = pipe.unet.dtype

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        height: int = 512,
        width: int = 512,
        controller: Optional[AttentionControl] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Image.Image, AttentionControl]:
        """
        Generate a single image while recording attention maps via controller.
        """
        seed_everything(seed)
        gen = get_generator(seed, self.device)

        if controller is None:
            from src.attention.store import AttentionStore
            controller = AttentionStore()

        controller.num_steps = num_inference_steps
        controller.reset()
        register_attention_control(self.pipe, controller)

        # Encode text
        uncond_emb = self.tokenizer.get_unconditional_embeddings()
        cond_emb, meta = self.tokenizer.encode(prompt)
        text_embeddings = torch.cat([uncond_emb, cond_emb]).to(device=self.device, dtype=self.dtype)

        # Initialize latent noise
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, height // 8, width // 8),
            generator=gen,
            device="cpu",
            dtype=self.dtype,
        ).to(self.device)

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            controller.step_callback()

            if callback:
                callback(i + 1, num_inference_steps)

        # Restore empty control
        register_attention_control(self.pipe, EmptyControl())

        images = latent_to_pil(latents, self.pipe.vae)
        return images[0], controller

    @torch.no_grad()
    def edit(
        self,
        source_prompt: str,
        target_prompt: str,
        controller: AttentionControl,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        height: int = 512,
        width: int = 512,
        initial_latents: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Image.Image, Image.Image, AttentionControl]:
        """
        Execute joint Prompt-to-Prompt diffusion.
        Simultaneously generates source and edited images while controlling cross- and self-attention.

        Returns:
            (original_image, edited_image, controller)
        """
        seed_everything(seed)
        gen = get_generator(seed, self.device)

        controller.num_steps = num_inference_steps
        controller.reset()
        register_attention_control(self.pipe, controller)

        # Encode prompts: [uncond_src, uncond_tgt, cond_src, cond_tgt]
        uncond_src = self.tokenizer.get_unconditional_embeddings()
        uncond_tgt = self.tokenizer.get_unconditional_embeddings()
        cond_src, _ = self.tokenizer.encode(source_prompt)
        cond_tgt, _ = self.tokenizer.encode(target_prompt)

        text_embeddings = torch.cat([uncond_src, uncond_tgt, cond_src, cond_tgt]).to(
            device=self.device, dtype=self.dtype
        )

        # Initialize latents: duplicated for source and target
        if initial_latents is None:
            init_noise = torch.randn(
                (1, self.pipe.unet.config.in_channels, height // 8, width // 8),
                generator=gen,
                device="cpu",
                dtype=self.dtype,
            ).to(self.device)
            # Duplicate for both paths
            latents = torch.cat([init_noise, init_noise])
        else:
            latents = torch.cat([initial_latents, initial_latents]).to(
                device=self.device, dtype=self.dtype
            )

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # Batch shape: (4, 4, H//8, W//8) -> [uncond_src, uncond_tgt, cond_src, cond_tgt]
            latent_model_input = torch.cat([latents, latents])
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            # Separate unconditional and conditional predictions
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            controller.step_callback()

            if callback:
                callback(i + 1, num_inference_steps)

        # Restore pass-through control
        register_attention_control(self.pipe, EmptyControl())

        images = latent_to_pil(latents, self.pipe.vae)
        return images[0], images[1], controller
