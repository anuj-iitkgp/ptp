"""
Prompt-to-Prompt Word Swap Example.
Original: "A photo of a dog sitting on a beach"
Edited:   "A photo of a cat sitting on a beach"

Author: Anuj Yadav (IIT Kharagpur)
"""

import argparse
import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model
from src.pipeline.editing import edit_word_swap
from src.utils.image import create_side_by_side
from src.attention.visualization import (
    get_token_attention_map,
    create_attention_overlay,
    create_token_comparison_grid,
)
from src.evaluation.comparison import generate_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Prompt-to-Prompt Word Swap CLI Example")
    parser.add_argument("--source", type=str, default="A photo of a dog sitting on a beach")
    parser.add_argument("--target", type=str, default="A photo of a cat sitting on a beach")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cross-replace", type=float, default=0.8)
    parser.add_argument("--self-replace", type=float, default=0.4)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("PROMPT-TO-PROMPT: WORD SWAP")
    print(f"Source: '{args.source}'")
    print(f"Target: '{args.target}'")
    print(f"Seed: {args.seed} | Steps: {args.steps}")
    print("=" * 60)

    pipe = load_model()

    print("\nRunning joint Prompt-to-Prompt diffusion...")
    img_orig, img_edit, controller, alignment = edit_word_swap(
        pipe=pipe,
        source_prompt=args.source,
        target_prompt=args.target,
        cross_replace_steps=args.cross_replace,
        self_replace_steps=args.self_replace,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    # Save outputs
    path_orig = os.path.join(args.output_dir, "word_swap_original.png")
    path_edit = os.path.join(args.output_dir, "word_swap_edited.png")
    path_sbs = os.path.join(args.output_dir, "word_swap_comparison.png")

    img_orig.save(path_orig)
    img_edit.save(path_edit)

    sbs = create_side_by_side(img_orig, img_edit, label1="Original (Dog)", label2="Edited (Cat)")
    sbs.save(path_sbs)
    print(f"\nSaved original: {path_orig}")
    print(f"Saved edited:   {path_edit}")
    print(f"Saved side-by-side comparison: {path_sbs}")

    # Generate evaluation metrics
    report = generate_evaluation_report(img_orig, img_edit, args.source, args.target, pipe=pipe)
    print("\nQuantitative Evaluation:")
    for k, v in report["structural_fidelity"].items():
        print(f"  {k}: {v}")

    # Visualize attention if swapped tokens exist
    if alignment.get("changed_tokens"):
        ch = alignment["changed_tokens"][0]
        src_idx = ch["src_idx"]
        tgt_idx = ch["tgt_idx"]
        grid = create_token_comparison_grid(
            img_orig, img_edit, controller, src_idx, tgt_idx, ch["src"], ch["tgt"]
        )
        path_attn = os.path.join(args.output_dir, "word_swap_attention_grid.png")
        grid.save(path_attn)
        print(f"Saved attention heatmap grid: {path_attn}")

    print("\nDone! Prompt-to-Prompt Word Swap completed successfully.")


if __name__ == "__main__":
    main()
