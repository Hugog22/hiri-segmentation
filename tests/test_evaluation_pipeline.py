"""
Tests for error analysis, uncertainty, and benchmark evaluation modules.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    apply_noise_perturbation,
    bin_slice_thickness,
    find_worst_cases,
    stratify_by_metadata,
)
from src.evaluation.uncertainty import (
    compute_ensemble_variance,
    compute_entropy,
    extract_boundary_mask,
)


class TestErrorAnalysis:
    def test_find_worst_cases(self):
        df = pd.DataFrame({
            "case_id": [f"c_{i}" for i in range(10)],
            "dice": [0.95, 0.82, 0.60, 0.91, 0.75, 0.88, 0.50, 0.93, 0.79, 0.85]
        })
        worst = find_worst_cases(df, metric_col="dice", n_cases=3, ascending=True)
        assert len(worst) == 3
        # Worst case should be c_6 (dice = 0.50)
        assert worst.iloc[0]["case_id"] == "c_6"
        assert worst.iloc[0]["dice"] == 0.50

    def test_bin_slice_thickness(self):
        assert bin_slice_thickness(1.0) == "thin (<=1.5mm)"
        assert bin_slice_thickness(2.5) == "medium (1.5-3.0mm)"
        assert bin_slice_thickness(5.0) == "thick (>3.0mm)"

    def test_noise_perturbation_bounds(self):
        img = np.ones((10, 10, 10)) * 0.5
        noisy = apply_noise_perturbation(img, sigma=0.1)
        assert np.all(noisy >= 0.0)
        assert np.all(noisy <= 1.0)


class TestUncertainty:
    def test_entropy_computation(self):
        # 4 classes, 10x10x10 voxels
        # Uniform distribution across 4 classes: p = 0.25 -> maximum entropy = -4 * 0.25 * ln(0.25) = ln(4) approx 1.386
        probs = np.ones((4, 10, 10, 10), dtype=np.float32) * 0.25
        ent = compute_entropy(probs)
        assert np.allclose(ent, np.log(4), atol=1e-3)

        # Deterministic distribution: p = [1, 0, 0, 0] -> minimum entropy = 0
        det_probs = np.zeros((4, 10, 10, 10), dtype=np.float32)
        det_probs[0] = 1.0
        det_ent = compute_entropy(det_probs)
        assert np.allclose(det_ent, 0.0, atol=1e-3)

    def test_ensemble_variance(self):
        p1 = np.zeros((4, 8, 8, 8), dtype=np.float32)
        p1[1] = 1.0  # class 1
        p2 = np.zeros((4, 8, 8, 8), dtype=np.float32)
        p2[2] = 1.0  # class 2
        var = compute_ensemble_variance([p1, p2])
        # Variance should be > 0 where models disagree
        assert np.all(var > 0)

    def test_extract_boundary_mask(self):
        seg = np.zeros((20, 20, 20), dtype=np.uint8)
        seg[5:15, 5:15, 5:15] = 1  # 10x10x10 cube
        boundary = extract_boundary_mask(seg, radius=1)
        # Deep interior should not be in boundary
        assert boundary[10, 10, 10] == False
        # Surface voxels should be detected
        assert np.any(boundary)
