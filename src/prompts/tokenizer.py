"""
CLIP Tokenization and text embedding extraction utilities.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import List, Dict, Any, Tuple
import torch
from transformers import CLIPTokenizer, CLIPTextModel


class PromptTokenizer:
    """
    Wrapper around CLIPTokenizer and CLIPTextModel for token-level analysis.
    """

    def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = text_encoder.device

    def tokenize(self, prompt: str) -> Dict[str, Any]:
        """
        Tokenize a single text prompt and return detailed token metadata.

        Returns:
            Dict containing:
                - 'input_ids': torch.Tensor of shape (1, 77)
                - 'tokens': List of decoded token strings
                - 'token_ids': List of integer token IDs
                - 'clean_tokens': List of tokens with '</w>' cleaned for readable display
                - 'num_real_tokens': Number of tokens excluding padding (<|startoftext|>, <|endoftext|>)
        """
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids[0].tolist()

        raw_tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids]
        clean_tokens = [t.strip().replace("</w>", "") for t in raw_tokens]

        # Count tokens until <|endoftext|>
        end_token_id = self.tokenizer.eos_token_id
        try:
            end_pos = input_ids.index(end_token_id)
            num_real_tokens = end_pos + 1
        except ValueError:
            num_real_tokens = len(input_ids)

        return {
            "input_ids": inputs.input_ids.to(self.device),
            "tokens": raw_tokens,
            "clean_tokens": clean_tokens,
            "token_ids": input_ids,
            "num_real_tokens": num_real_tokens,
            "prompt": prompt,
        }

    def encode(self, prompt: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode a prompt into text embeddings.
        Returns:
            - embeddings: torch.Tensor of shape (1, 77, 768)
            - metadata: Dict from tokenize()
        """
        meta = self.tokenize(prompt)
        with torch.no_grad():
            embeddings = self.text_encoder(meta["input_ids"])[0]
        return embeddings, meta

    def get_unconditional_embeddings(self) -> torch.Tensor:
        """Encode unconditional null text ('') for Classifier-Free Guidance."""
        emb, _ = self.encode("")
        return emb
