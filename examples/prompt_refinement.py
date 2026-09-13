"""
Prompt-to-Prompt Prompt Refinement Example.
Original: "A dog in a park"
Edited:   "A small golden dog in a park"

Author: Anuj Yadav (IIT Kharagpur)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model
from src.pipeline.editing import edit_prompt_refinement
from src.utils.image import create_side_by_side
from src.evaluation.comparison import generate_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Prompt-to-Prompt Refinement CLI Example")
    parser.add_argument("--source", type=str, default="A dog in a park")
    parser.add_argument("--target", type=str, default="A small golden dog in a park")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("PROMPT-TO-PROMPT: PROMPT REFINEMENT")
    print(f"Source: '{args.source}'")
    print(f"Target: '{args.target}'")
    print(f"Seed: {args.seed} | Steps: {args.steps}")
    print("=" * 60)

    pipe = load_model()

    print("\nRunning Prompt Refinement diffusion...")
    img_orig, img_edit, controller, alignment = edit_prompt_refinement(
        pipe=pipe,
        source_prompt=args.source,
        target_prompt=args.target,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    path_sbs = os.path.join(args.output_dir, "refinement_comparison.png")
    sbs = create_side_by_side(img_orig, img_edit, label1="Original", label2="Refined (Small Golden Dog)")
    sbs.save(path_sbs)
    print(f"\nSaved refinement side-by-side comparison: {path_sbs}")

    report = generate_evaluation_report(img_orig, img_edit, args.source, args.target, pipe=pipe)
    print("\nQuantitative Evaluation:")
    for k, v in report["structural_fidelity"].items():
        print(f"  {k}: {v}")

    print("\nDone! Prompt Refinement completed successfully.")


if __name__ == "__main__":
    main()
