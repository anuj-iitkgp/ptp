"""
Evaluation package for Prompt-to-Prompt editing.
"""

from src.evaluation.metrics import (
    compute_structure_metrics,
    compute_clip_directional_similarity,
)
from src.evaluation.comparison import generate_evaluation_report

__all__ = [
    "compute_structure_metrics",
    "compute_clip_directional_similarity",
    "generate_evaluation_report",
]
