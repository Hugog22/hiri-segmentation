"""
Module for computing organ volumes from segmentation masks.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_original_spacing(json_path: Path) -> tuple[float, float, float]:
    """Load the original spacing saved during preprocessing."""
    if not json_path.exists():
        raise FileNotFoundError(f"Spacing JSON not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return tuple(data.get("spacing", [1.0, 1.0, 1.0]))

def compute_volume_ml(mask_path: Path, label: int, spacing: tuple[float, float, float] | None = None) -> float:
    """
    Compute volume for a given label in mL.
    If spacing is None, read from NIfTI header.
    """
    img = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(img)
    
    if spacing is None:
        spacing = img.GetSpacing()
    else:
        logger.warning("Using provided original spacing instead of image header spacing.")
        
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_count = np.sum(mask_array == label)
    
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0
    return float(volume_ml)

def compute_all_volumes(mask_path: Path, label_map: dict[str, int], original_spacing_json: Path | None = None) -> dict[str, float]:
    """
    Compute volumes for all organs in the label_map.
    """
    spacing = None
    if original_spacing_json and original_spacing_json.exists():
        spacing = load_original_spacing(original_spacing_json)
        
    volumes = {}
    for organ_name, label_idx in label_map.items():
        vol = compute_volume_ml(mask_path, label_idx, spacing)
        volumes[organ_name] = vol
        
    return volumes

def compute_volume_error(pred_volumes: dict[str, float], gt_volumes: dict[str, float]) -> dict[str, dict[str, float]]:
    """
    Compute absolute and relative volume errors.
    """
    errors = {}
    for organ in pred_volumes:
        if organ in gt_volumes:
            pred = pred_volumes[organ]
            gt = gt_volumes[organ]
            abs_err = abs(pred - gt)
            rel_err = (abs_err / gt * 100.0) if gt > 0 else 0.0
            
            errors[organ] = {
                "absolute_error_ml": abs_err,
                "relative_error_percent": rel_err
            }
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute organ volumes from masks")
    parser.add_argument("--mask-path", type=str, required=True, help="Path to segmentation mask NIfTI")
    parser.add_argument("--label-map", type=str, required=True, help="JSON string of label map e.g. '{\"liver\":1}'")
    parser.add_argument("--original-spacing-json", type=str, help="Path to JSON with original spacing")
    parser.add_argument("--output-csv", type=str, help="Path to output CSV")
    
    args = parser.parse_args()
    
    label_map = json.loads(args.label_map)
    vols = compute_all_volumes(
        Path(args.mask_path), 
        label_map, 
        Path(args.original_spacing_json) if args.original_spacing_json else None
    )
    
    logger.info(f"Volumes: {vols}")
    
    if args.output_csv:
        df = pd.DataFrame([vols])
        df.to_csv(args.output_csv, index=False)
