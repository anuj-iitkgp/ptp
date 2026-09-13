"""
Latent and Null-Text Inversion module for real image Prompt-to-Prompt editing.

Background:
Standard text-to-image generation samples Gaussian noise z_T ~ N(0, I) and denoises
it along an ODE trajectory to z_0. For editing a REAL image, we must invert the image
x_0 into its latent code z_T such that regenerating from z_T reconstructs x_0.

Methods implemented:
1. DDIM Inversion: Deterministic reversal of the DDIM sampling ODE.
2. Null-Text Inversion (Mokady et al., 2022): Optimizes the unconditional null-text
   embedding at each timestep to counteract CFG trajectory drift, allowing
   high-fidelity reconstruction and attention editing of real photos.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import List, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

from src.utils.image import pil_to_latent
from src.prompts.tokenizer import PromptTokenizer


class DDIMInversion:
    """
    Deterministic DDIM Inversion using reversed ODE dynamics.
    """

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.tokenizer = PromptTokenizer(pipe.tokenizer, pipe.text_encoder)
        self.device = pipe.device
        self.dtype = pipe.unet.dtype

    @torch.no_grad()
    def invert(
        self,
        image: Image.Image,
        prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Invert a real PIL image into latent noise z_T.

        Args:
            image: Input RGB image.
            prompt: Text description of the input image (can be empty string).
            num_inference_steps: Number of inversion steps.
            guidance_scale: Guidance scale during inversion (1.0 recommended for DDIM inversion).

        Returns:
            z_T: The recovered noise latent at step T.
            trajectory: List of latents [z_0, z_1, ..., z_T].
        """
        # Encode real image to initial latent z_0
        z_0 = pil_to_latent(image, self.pipe.vae, self.device, dtype=self.dtype)

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = reversed(self.pipe.scheduler.timesteps)

        # Encode text
        cond_emb, _ = self.tokenizer.encode(prompt)
        text_embeddings = cond_emb.to(device=self.device, dtype=self.dtype)

        latents = z_0
        trajectory = [latents]

        for i, t in enumerate(timesteps):
            latent_model_input = self.pipe.scheduler.scale_model_input(latents, t)

            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            # Reversed DDIM step: z_{t+1} from z_t
            # In diffusers DDIM, previous timestep in reversed direction is next forward step
            prev_timestep = (
                t - self.pipe.scheduler.config.num_train_timesteps // num_inference_steps
            )

            alpha_prod_t = self.pipe.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.pipe.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.pipe.scheduler.final_alpha_cumprod
            )

            beta_prod_t = 1 - alpha_prod_t

            # Deterministic ODE step formula
            latents = (
                (latents - beta_prod_t ** 0.5 * noise_pred)
                * (alpha_prod_t_prev ** 0.5 / alpha_prod_t ** 0.5)
                + (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
            )

            trajectory.append(latents)
            if callback:
                callback(i + 1, num_inference_steps)

        z_T = trajectory[-1]
        return z_T, trajectory


class NullTextInversion:
    """
    Null-Text Inversion for editing real images with high Classifier-Free Guidance.
    Optimizes the unconditional text embedding per timestep.
    """

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.ddim_inv = DDIMInversion(pipe)
        self.tokenizer = PromptTokenizer(pipe.tokenizer, pipe.text_encoder)
        self.device = pipe.device
        self.dtype = pipe.unet.dtype

    def invert(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_inner_steps: int = 5,
        learning_rate: float = 1e-2,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Execute Null-Text Inversion.
        First collects the DDIM inversion trajectory, then optimizes the null-text embedding
        at each timestep to minimize reconstruction divergence under CFG.
        """
        # Step 1: Recover DDIM forward trajectory
        z_T, trajectory = self.ddim_inv.invert(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
        )

        uncond_emb_base = self.tokenizer.get_unconditional_embeddings().to(
            device=self.device, dtype=self.dtype
        )
        cond_emb, _ = self.tokenizer.encode(prompt)
        cond_emb = cond_emb.to(device=self.device, dtype=self.dtype)

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        uncond_embeddings_list = []
        cur_latents = z_T.clone()

        # Step 2: Optimize null-text embedding per timestep
        for i, t in enumerate(timesteps):
            target_latent = trajectory[-(i + 2)]  # Expected latent at next step

            uncond_emb = uncond_emb_base.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([uncond_emb], lr=learning_rate)

            for inner in range(num_inner_steps):
                optimizer.zero_grad()
                text_embeddings = torch.cat([uncond_emb, cond_emb])
                latent_input = torch.cat([cur_latents] * 2)

                noise_pred = self.pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                pred_prev = self.pipe.scheduler.step(noise_pred, t, cur_latents).prev_sample
                loss = F.mse_loss(pred_prev, target_latent)
                loss.backward()
                optimizer.step()

            uncond_embeddings_list.append(uncond_emb.detach())
            with torch.no_grad():
                text_embeddings = torch.cat([uncond_emb.detach(), cond_emb])
                latent_input = torch.cat([cur_latents] * 2)
                noise_pred = self.pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                cur_latents = self.pipe.scheduler.step(noise_pred, t, cur_latents).prev_sample

            if callback:
                callback(i + 1, num_inference_steps)

        return z_T, uncond_embeddings_list
