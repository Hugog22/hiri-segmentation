"""
Postprocessing module for segmentations (connected components, hole filling).
"""
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import yaml

logger = logging.getLogger(__name__)

def keep_largest_component(binary_mask: np.ndarray, n_keep: int = 1) -> np.ndarray:
    """Keep the N largest 3D connected components."""
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return binary_mask
        
    component_sizes = np.bincount(labeled_mask.ravel())
    # ignore background (index 0)
    component_sizes[0] = 0
    
    # Get indices of largest components
    largest_components = np.argsort(component_sizes)[::-1][:n_keep]
    
    output_mask = np.zeros_like(binary_mask)
    for comp_idx in largest_components:
        if component_sizes[comp_idx] > 0:
            output_mask[labeled_mask == comp_idx] = 1
            
    return output_mask

def remove_small_components(binary_mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Remove components smaller than threshold."""
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return binary_mask
        
    component_sizes = np.bincount(labeled_mask.ravel())
    
    # Find components to remove
    too_small = component_sizes < min_voxels
    too_small_mask = too_small[labeled_mask]
    
    output_mask = binary_mask.copy()
    output_mask[too_small_mask] = 0
    
    return output_mask

def fill_holes_3d(binary_mask: np.ndarray) -> np.ndarray:
    """Fill internal holes using binary_fill_holes."""
    return ndimage.binary_fill_holes(binary_mask).astype(binary_mask.dtype)

def compute_component_volumes(binary_mask: np.ndarray, spacing: Tuple[float, float, float]) -> List[float]:
    """Return volumes in mL of each connected component, sorted descending."""
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return []
        
    voxel_volume_mm3 = float(np.prod(spacing))
    component_sizes = np.bincount(labeled_mask.ravel())[1:] # skip background
    
    volumes_ml = [(size * voxel_volume_mm3) / 1000.0 for size in component_sizes if size > 0]
    return sorted(volumes_ml, reverse=True)

def postprocess_segmentation(segmentation: np.ndarray, spacing: Tuple[float, float, float], config: Dict[str, Any]) -> np.ndarray:
    """
    Main postprocessing pipeline.
    """
    post_cfg = config.get("inference", {}).get("postprocessing", {})
    min_volume_threshold = post_cfg.get("min_volume_threshold", 100)
    
    out_seg = np.zeros_like(segmentation)
    
    # 1. Liver (label 1): keep only LARGEST connected component
    liver_mask = (segmentation == 1).astype(np.uint8)
    if np.any(liver_mask):
        liver_mask = fill_holes_3d(liver_mask)
        liver_mask = keep_largest_component(liver_mask, n_keep=1)
        liver_mask = remove_small_components(liver_mask, min_volume_threshold)
        out_seg[liver_mask == 1] = 1
        
    # 2. Kidneys (labels 2, 3): keep the largest component for each
    for kidney_label, name in [(2, "Right"), (3, "Left")]:
        kidney_mask = (segmentation == kidney_label).astype(np.uint8)
        if np.any(kidney_mask):
            kidney_mask = fill_holes_3d(kidney_mask)
            kidney_mask = keep_largest_component(kidney_mask, n_keep=1)
            kidney_mask = remove_small_components(kidney_mask, min_volume_threshold)
            
            # Check suspicious small volume
            volumes = compute_component_volumes(kidney_mask, spacing)
            if volumes and volumes[0] < 10.0:
                logger.warning(f"Suspiciously small {name} kidney detected: {volumes[0]:.2f} mL (<10 mL). Retaining but flagged.")
                
            out_seg[kidney_mask == 1] = kidney_label
            
    return out_seg

def postprocess_batch(input_dir: Path, output_dir: Path, config: Dict[str, Any]) -> None:
    """Process all segmentation NIfTIs in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for nifti_file in input_dir.glob("*.nii.gz"):
        logger.info(f"Postprocessing {nifti_file.name}")
        img = nib.load(str(nifti_file))
        data = img.get_fdata()
        spacing = tuple(img.header.get_zooms()[:3])
        
        post_data = postprocess_segmentation(data, spacing, config)
        
        out_nifti = nib.Nifti1Image(post_data.astype(np.uint8), img.affine, img.header)
        nib.save(out_nifti, str(output_dir / nifti_file.name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess segmentations.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    postprocess_batch(args.input_dir, args.output_dir, cfg)
