"""
Uncertainty estimation module for 3D medical image segmentation.

Computes:
- Voxel-wise predictive entropy: H(x) = - sum_c p_c * log(p_c + eps)
- Ensemble / TTA variance: Var(p_c) across M stochastic passes or ensemble members.
- Organ-level and boundary uncertainty scores to identify doubtful cases for radiologist review.
"""
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ORGAN_LABELS = {
    "liver": 1,
    "right_kidney": 2,
    "left_kidney": 3
}


def compute_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute voxel-wise predictive entropy across classes.
    probs: shape (C, D, H, W) or (B, C, D, H, W) where C is number of classes.
    Returns: entropy map of shape (D, H, W) or (B, D, H, W).
    """
    if probs.ndim == 5:
        probs = probs[0]
    
    # Clip for numerical stability
    p = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=0)
    return entropy.astype(np.float32)


def compute_ensemble_variance(prob_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute voxel-wise variance across ensemble members or TTA predictions.
    prob_list: list of M probability maps, each shape (C, D, H, W).
    Returns: mean variance across foreground classes of shape (D, H, W).
    """
    if not prob_list:
        raise ValueError("prob_list cannot be empty")
        
    stack = np.stack(prob_list, axis=0)  # (M, C, D, H, W)
    # Variance along ensemble axis M
    var_per_class = np.var(stack, axis=0)  # (C, D, H, W)
    # Average variance across foreground classes (1, 2, 3)
    fg_variance = np.mean(var_per_class[1:], axis=0)  # (D, H, W)
    return fg_variance.astype(np.float32)


def extract_boundary_mask(segmentation: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Extract boundary band around segmented structures.
    """
    fg = (segmentation > 0).astype(bool)
    dilated = binary_dilation(fg, iterations=radius)
    eroded = ~binary_dilation(~fg, iterations=radius)
    boundary = np.logical_xor(dilated, eroded)
    return boundary


def compute_case_uncertainty_metrics(
    probs: np.ndarray,
    segmentation: np.ndarray,
    ensemble_prob_list: Optional[List[np.ndarray]] = None
) -> Dict[str, float]:
    """
    Calculate summary uncertainty scores for a 3D scan:
    - Global mean entropy inside foreground
    - Boundary entropy (entropy specifically on organ surfaces)
    - Ensemble variance (if multiple predictions provided)
    - Per-organ mean entropy
    """
    entropy = compute_entropy(probs)
    fg_mask = (segmentation > 0)
    boundary_mask = extract_boundary_mask(segmentation)
    
    results: Dict[str, float] = {}
    
    # Global foreground entropy
    if np.any(fg_mask):
        results["mean_fg_entropy"] = float(np.mean(entropy[fg_mask]))
        results["max_fg_entropy"] = float(np.max(entropy[fg_mask]))
    else:
        results["mean_fg_entropy"] = 0.0
        results["max_fg_entropy"] = 0.0
        
    # Boundary uncertainty
    if np.any(boundary_mask):
        results["mean_boundary_entropy"] = float(np.mean(entropy[boundary_mask]))
    else:
        results["mean_boundary_entropy"] = 0.0
        
    # Per-organ entropy
    for organ_name, lbl in ORGAN_LABELS.items():
        organ_mask = (segmentation == lbl)
        if np.any(organ_mask):
            results[f"mean_{organ_name}_entropy"] = float(np.mean(entropy[organ_mask]))
        else:
            results[f"mean_{organ_name}_entropy"] = 0.0
            
    # Ensemble variance if available
    if ensemble_prob_list and len(ensemble_prob_list) > 1:
        variance = compute_ensemble_variance(ensemble_prob_list)
        if np.any(fg_mask):
            results["mean_fg_variance"] = float(np.mean(variance[fg_mask]))
            results["mean_boundary_variance"] = float(np.mean(variance[boundary_mask])) if np.any(boundary_mask) else 0.0
            
    # Ambiguity flag: high boundary uncertainty suggests contour ambiguity
    # Empirically, normalized binary entropy > 0.35 on boundary flags challenging cases
    results["is_ambiguous_case"] = float(results["mean_boundary_entropy"] > 0.35)
    
    return results


def save_uncertainty_maps(
    entropy: np.ndarray,
    variance: Optional[np.ndarray],
    ref_nii_path: Path,
    output_dir: Path,
    case_id: str
) -> None:
    """Save 3D NIfTI volumes of voxel-wise uncertainty."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_nii = nib.load(str(ref_nii_path))
    
    entropy_nii = nib.Nifti1Image(entropy, ref_nii.affine, ref_nii.header)
    nib.save(entropy_nii, str(output_dir / f"{case_id}_entropy.nii.gz"))
    
    if variance is not None:
        var_nii = nib.Nifti1Image(variance, ref_nii.affine, ref_nii.header)
        nib.save(var_nii, str(output_dir / f"{case_id}_variance.nii.gz"))
    logger.info(f"Saved uncertainty volumes for case {case_id} in {output_dir}")
