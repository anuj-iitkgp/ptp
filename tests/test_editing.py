"""
Unit tests for evaluation metrics and pipeline execution components.

Author: Anuj Yadav (IIT Kharagpur)
"""

import pytest
import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_structure_metrics
from src.evaluation.comparison import generate_evaluation_report


def test_structure_metrics_identical_images():
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img1 = Image.fromarray(arr)
    img2 = Image.fromarray(arr)

    metrics = compute_structure_metrics(img1, img2)
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["l1_diff"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["edge_similarity"] == pytest.approx(1.0, abs=1e-3)


def test_structure_metrics_dissimilar_images():
    img1 = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.ones((128, 128, 3), dtype=np.uint8) * 255)

    metrics = compute_structure_metrics(img1, img2)
    assert metrics["ssim"] < 0.1
    assert metrics["l1_diff"] == pytest.approx(1.0, abs=1e-3)


def test_evaluation_report_generation():
    img1 = Image.new("RGB", (64, 64), color="red")
    img2 = Image.new("RGB", (64, 64), color="blue")

    report = generate_evaluation_report(
        img1, img2, "a red car", "a blue car"
    )

    assert "structural_fidelity" in report
    assert "ssim" in report["structural_fidelity"]
    assert "summary" in report
