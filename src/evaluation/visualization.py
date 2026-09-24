"""
Visualization module for qualitative 2D and 3D segmentation evaluation.

Generates:
- 2D Orthogonal slice overlays (axial, coronal, sagittal) showing GT vs prediction contours.
- Categorical error maps: True Positive (green), False Positive (red), False Negative (blue).
- Comparative figures comparing multiple models on the same anatomical slice.
"""
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ORGAN_NAMES = {
    1: "Liver",
    2: "Right Kidney",
    3: "Left Kidney"
}

# Anatomical color palette
ORGAN_COLORS = {
    1: "#e41a1c",  # Liver: Red
    2: "#377eb8",  # Right Kidney: Blue
    3: "#4daf4a"   # Left Kidney: Green
}


def find_best_slices(mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Find slices with maximum organ cross-sectional area in (axial, coronal, sagittal).
    mask is assumed to be in RAS orientation:
    - axis 0: Sagittal (Right-Left)
    - axis 1: Coronal (Anterior-Posterior)
    - axis 2: Axial (Superior-Inferior)
    """
    fg = (mask > 0).astype(int)
    if not np.any(fg):
        return mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 2
        
    sag_slice = int(np.argmax(np.sum(fg, axis=(1, 2))))
    cor_slice = int(np.argmax(np.sum(fg, axis=(0, 2))))
    ax_slice = int(np.argmax(np.sum(fg, axis=(0, 1))))
    return sag_slice, cor_slice, ax_slice


def create_error_overlay(
    image_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    target_label: int,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create RGB overlay:
    - True Positive (TP): Green
    - False Positive (FP): Red
    - False Negative (FN): Blue
    """
    # Normalize image to [0, 1]
    img_norm = np.clip((image_slice - np.percentile(image_slice, 1)) / 
                       (np.percentile(image_slice, 99) - np.percentile(image_slice, 1) + 1e-8), 0, 1)
    rgb = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
    
    gt_binary = (gt_slice == target_label)
    pred_binary = (pred_slice == target_label)
    
    tp = np.logical_and(gt_binary, pred_binary)
    fp = np.logical_and(~gt_binary, pred_binary)
    fn = np.logical_and(gt_binary, ~pred_binary)
    
    # TP: Green [0, 1, 0]
    rgb[tp] = (1 - alpha) * rgb[tp] + alpha * np.array([0.0, 1.0, 0.0])
    # FP: Red [1, 0, 0]
    rgb[fp] = (1 - alpha) * rgb[fp] + alpha * np.array([1.0, 0.0, 0.0])
    # FN: Blue [0, 0.4, 1]
    rgb[fn] = (1 - alpha) * rgb[fn] + alpha * np.array([0.0, 0.4, 1.0])
    
    return rgb


def plot_orthogonal_comparison(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    case_id: str,
    output_path: Path,
    hu_window: Tuple[float, float] = (-150.0, 250.0),
    title_suffix: str = ""
) -> None:
    """
    Generate 3x3 grid:
    Rows: Sagittal, Coronal, Axial
    Cols: Raw Image with HU window, Ground Truth Overlays, Prediction Overlays with TP/FP/FN
    """
    # Apply HU window
    clipped_img = np.clip(image, hu_window[0], hu_window[1])
    
    sag_idx, cor_idx, ax_idx = find_best_slices(gt)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    views = [
        ("Sagittal (R-L)", np.rot90(clipped_img[sag_idx, :, :]), np.rot90(gt[sag_idx, :, :]), np.rot90(pred[sag_idx, :, :])),
        ("Coronal (A-P)", np.rot90(clipped_img[:, cor_idx, :]), np.rot90(gt[:, cor_idx, :]), np.rot90(pred[:, cor_idx, :])),
        ("Axial (S-I)", np.rot90(clipped_img[:, :, ax_idx]), np.rot90(gt[:, :, ax_idx]), np.rot90(pred[:, :, ax_idx]))
    ]
    
    col_titles = ["CT Image (Windowed)", "Ground Truth", "Prediction (TP=Grn, FP=Red, FN=Blu)"]
    
    for row_idx, (view_name, img_slice, gt_slice, pred_slice) in enumerate(views):
        # Col 0: Raw image
        axes[row_idx, 0].imshow(img_slice, cmap="gray")
        axes[row_idx, 0].set_ylabel(view_name, fontsize=12, fontweight="bold")
        axes[row_idx, 0].axis("off")
        
        # Col 1: Ground truth contours/colors
        axes[row_idx, 1].imshow(img_slice, cmap="gray")
        for lbl, color in ORGAN_COLORS.items():
            mask_lbl = (gt_slice == lbl)
            if np.any(mask_lbl):
                axes[row_idx, 1].contour(mask_lbl, colors=[color], linewidths=1.5)
        axes[row_idx, 1].axis("off")
        
        # Col 2: Error overlay for all organs combined
        rgb_overlay = np.zeros((*img_slice.shape, 3))
        # Base grayscale
        img_norm = (img_slice - hu_window[0]) / (hu_window[1] - hu_window[0] + 1e-8)
        img_norm = np.clip(img_norm, 0, 1)
        rgb_overlay = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
        
        tp = np.logical_and(gt_slice > 0, pred_slice > 0)
        fp = np.logical_and(gt_slice == 0, pred_slice > 0)
        fn = np.logical_and(gt_slice > 0, pred_slice == 0)
        
        rgb_overlay[tp] = 0.5 * rgb_overlay[tp] + 0.5 * np.array([0.0, 1.0, 0.0])
        rgb_overlay[fp] = 0.5 * rgb_overlay[fp] + 0.5 * np.array([1.0, 0.0, 0.0])
        rgb_overlay[fn] = 0.5 * rgb_overlay[fn] + 0.5 * np.array([0.2, 0.5, 1.0])
        
        axes[row_idx, 2].imshow(rgb_overlay)
        axes[row_idx, 2].axis("off")
        
        if row_idx == 0:
            for c_idx, title in enumerate(col_titles):
                axes[0, c_idx].set_title(title, fontsize=13, fontweight="bold", pad=10)
                
    plt.suptitle(f"Case: {case_id} {title_suffix}", fontsize=16, fontweight="bold", y=0.94)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visualization to {output_path}")


def plot_model_comparison_slice(
    image_slice: np.ndarray,
    gt_slice: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    case_id: str,
    output_path: Path,
    hu_window: Tuple[float, float] = (-150.0, 250.0)
) -> None:
    """
    Compare multiple models side-by-side on the same axial slice.
    """
    n_models = len(model_predictions)
    fig, axes = plt.subplots(1, n_models + 2, figsize=(4 * (n_models + 2), 4.5))
    
    img_norm = np.clip((image_slice - hu_window[0]) / (hu_window[1] - hu_window[0] + 1e-8), 0, 1)
    
    # 1. CT
    axes[0].imshow(np.rot90(img_norm), cmap="gray")
    axes[0].set_title("CT Image", fontweight="bold")
    axes[0].axis("off")
    
    # 2. Ground Truth
    axes[1].imshow(np.rot90(img_norm), cmap="gray")
    for lbl, color in ORGAN_COLORS.items():
        m = (gt_slice == lbl)
        if np.any(m):
            axes[1].contour(np.rot90(m), colors=[color], linewidths=1.5)
    axes[1].set_title("Ground Truth", fontweight="bold")
    axes[1].axis("off")
    
    # 3. Models
    for idx, (m_name, pred_slice) in enumerate(model_predictions.items()):
        ax = axes[idx + 2]
        ax.imshow(np.rot90(img_norm), cmap="gray")
        for lbl, color in ORGAN_COLORS.items():
            m = (pred_slice == lbl)
            if np.any(m):
                ax.contour(np.rot90(m), colors=[color], linewidths=1.5)
        ax.set_title(m_name, fontweight="bold")
        ax.axis("off")
        
    plt.suptitle(f"Model Comparison — Case {case_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison to {output_path}")
