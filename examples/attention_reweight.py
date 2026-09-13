"""
Prompt-to-Prompt Attention Re-weighting Example.
Original: "A portrait of a woman wearing a hat"
Edited:   "A portrait of a woman wearing a hat" with hat weight 2.0x

Author: Anuj Yadav (IIT Kharagpur)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model
from src.pipeline.editing import edit_reweight
from src.utils.image import create_side_by_side
from src.evaluation.comparison import generate_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Prompt-to-Prompt Attention Re-weighting CLI Example")
    parser.add_argument("--prompt", type=str, default="A portrait of a woman wearing a hat")
    parser.add_argument("--word", type=str, default="hat")
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("PROMPT-TO-PROMPT: ATTENTION RE-WEIGHTING")
    print(f"Prompt: '{args.prompt}'")
    print(f"Target word: '{args.word}' | Multiplier: {args.weight}x")
    print(f"Seed: {args.seed} | Steps: {args.steps}")
    print("=" * 60)

    pipe = load_model()

    print("\nRunning Attention Re-weighting diffusion...")
    img_orig, img_edit, controller, alignment = edit_reweight(
        pipe=pipe,
        prompt=args.prompt,
        weights_dict={args.word: args.weight},
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    path_sbs = os.path.join(args.output_dir, "reweight_comparison.png")
    sbs = create_side_by_side(
        img_orig, img_edit, label1="Standard", label2=f"Amplified ({args.word}: {args.weight}x)"
    )
    sbs.save(path_sbs)
    print(f"\nSaved re-weighting side-by-side comparison: {path_sbs}")

    report = generate_evaluation_report(img_orig, img_edit, args.prompt, args.prompt, pipe=pipe)
    print("\nQuantitative Evaluation:")
    for k, v in report["structural_fidelity"].items():
        print(f"  {k}: {v}")

    print("\nDone! Attention Re-weighting completed successfully.")


if __name__ == "__main__":
    main()
