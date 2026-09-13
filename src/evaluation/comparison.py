"""
Evaluation report generation and side-by-side comparison tables.

Author: Anuj Yadav (IIT Kharagpur)
"""

from typing import Dict, Any
from PIL import Image
from src.evaluation.metrics import compute_structure_metrics, compute_clip_directional_similarity


def generate_evaluation_report(
    img_orig: Image.Image,
    img_edit: Image.Image,
    src_prompt: str,
    tgt_prompt: str,
    pipe=None,
) -> Dict[str, Any]:
    """
    Generate a complete quantitative evaluation report for an edit operation.
    """
    struct_metrics = compute_structure_metrics(img_orig, img_edit)

    report = {
        "source_prompt": src_prompt,
        "target_prompt": tgt_prompt,
        "structural_fidelity": struct_metrics,
        "summary": (
            f"SSIM: {struct_metrics['ssim']} | "
            f"PSNR: {struct_metrics['psnr_db']} dB | "
            f"Edge Similarity: {struct_metrics['edge_similarity']}"
        ),
    }

    if pipe is not None:
        edit_metrics = compute_clip_directional_similarity(
            img_orig, img_edit, src_prompt, tgt_prompt, pipe
        )
        report["edit_dynamics"] = edit_metrics

    return report
