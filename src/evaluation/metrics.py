"""
Comprehensive metrics module for segmentation evaluation.
Includes Dice, IoU, Hausdorff Distance (95%), and Average Surface Distance.
"""

import logging
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Compute standard Dice score."""
    p = (pred == label)
    g = (gt == label)
    
    p_sum = p.sum()
    g_sum = g.sum()
    
    if p_sum == 0 and g_sum == 0:
        return 1.0
        
    intersection = np.logical_and(p, g).sum()
    return float(2.0 * intersection / (p_sum + g_sum))

def iou_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Compute Intersection over Union (Jaccard) score."""
    p = (pred == label)
    g = (gt == label)
    
    p_sum = p.sum()
    g_sum = g.sum()
    
    if p_sum == 0 and g_sum == 0:
        return 1.0
        
    intersection = np.logical_and(p, g).sum()
    union = p_sum + g_sum - intersection
    if union == 0:
        return 1.0
    return float(intersection / union)

def get_surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Helper for distances using scipy."""
    # Get boundaries
    pred_border = np.logical_xor(pred_mask, binary_erosion(pred_mask))
    gt_border = np.logical_xor(gt_mask, binary_erosion(gt_mask))
    
    # Distance transforms
    if gt_border.any():
        dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
        dists_pred_to_gt = dt_gt[pred_border]
    else:
        dists_pred_to_gt = np.array([])
        
    if pred_border.any():
        dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
        dists_gt_to_pred = dt_pred[gt_border]
    else:
        dists_gt_to_pred = np.array([])
        
    return dists_pred_to_gt, dists_gt_to_pred

def hausdorff_95(pred: np.ndarray, gt: np.ndarray, label: int, spacing: tuple) -> float:
    """Compute 95th percentile Hausdorff distance."""
    p = (pred == label)
    g = (gt == label)
    
    if p.sum() == 0 or g.sum() == 0:
        logger.warning("Empty mask encountered for Hausdorff calculation.")
        return np.nan
        
    try:
        d_p2g, d_g2p = get_surface_distances(p, g, spacing)
        if len(d_p2g) == 0 or len(d_g2p) == 0:
            return np.nan
        hd95 = max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))
        return float(hd95)
    except Exception as e:
        logger.error(f"Error computing HD95: {e}")
        return np.nan

def average_surface_distance(pred: np.ndarray, gt: np.ndarray, label: int, spacing: tuple) -> float:
    """Compute mean symmetric surface distance."""
    p = (pred == label)
    g = (gt == label)
    
    if p.sum() == 0 or g.sum() == 0:
        logger.warning("Empty mask encountered for ASD calculation.")
        return np.nan
        
    try:
        d_p2g, d_g2p = get_surface_distances(p, g, spacing)
        if len(d_p2g) == 0 or len(d_g2p) == 0:
            return np.nan
        asd = (np.sum(d_p2g) + np.sum(d_g2p)) / (len(d_p2g) + len(d_g2p))
        return float(asd)
    except Exception as e:
        logger.error(f"Error computing ASD: {e}")
        return np.nan

def compute_all_metrics(pred_path: Path, gt_path: Path, label_map: dict, spacing: tuple | None = None) -> dict:
    """Compute all metrics for a given prediction and ground truth pair."""
    img_pred = nib.load(str(pred_path))
    img_gt = nib.load(str(gt_path))
    
    pred_data = img_pred.get_fdata()
    gt_data = img_gt.get_fdata()
    
    if spacing is None:
        spacing = img_gt.header.get_zooms()[:3]
        
    results = {}
    for organ, lbl in label_map.items():
        results[organ] = {
            "dice": dice_score(pred_data, gt_data, lbl),
            "iou": iou_score(pred_data, gt_data, lbl),
            "hd95": hausdorff_95(pred_data, gt_data, lbl, spacing),
            "asd": average_surface_distance(pred_data, gt_data, lbl, spacing)
        }
    return results

def aggregate_metrics(results: list[dict]) -> pd.DataFrame:
    """Aggregate case-level results into summary statistics."""
    records = []
    for r in results:
        case_id = r.get('case_id', 'unknown')
        for k, v in r.items():
            if isinstance(v, dict):
                row = {'case_id': case_id, 'organ': k}
                row.update(v)
                records.append(row)
                
    df = pd.DataFrame(records)
    if df.empty:
        return df
        
    agg = df.groupby('organ').agg(
        dice_mean=('dice', 'mean'),
        dice_std=('dice', 'std'),
        dice_median=('dice', 'median'),
        hd95_mean=('hd95', 'mean'),
        hd95_median=('hd95', 'median'),
        asd_mean=('asd', 'mean')
    ).reset_index()
    return agg
