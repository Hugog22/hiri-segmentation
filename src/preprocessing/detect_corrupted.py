"""
Module to detect corrupted or invalid medical images.
"""
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_case(case_dir: Path) -> dict:
    img_path = case_dir / 'image.nii.gz'
    mask_path = case_dir / 'unified_label.nii.gz'
    if not mask_path.exists():
        mask_path = case_dir / 'label.nii.gz'
        
    results = {'case_id': case_dir.name, 'pass': True}
    
    try:
        if not img_path.exists():
            raise FileNotFoundError("Image not found")
        img_nii = nib.load(str(img_path))
        img_data = img_nii.get_fdata()
        
        # NaN / Inf
        if np.isnan(img_data).any() or np.isinf(img_data).any():
            results['no_nan_inf'] = False
            results['pass'] = False
        else:
            results['no_nan_inf'] = True
            
        # 3D
        if len(img_data.shape) != 3:
            results['is_3d'] = False
            results['pass'] = False
        else:
            results['is_3d'] = True
            
        # Spacing
        spacing = img_nii.header.get_zooms()
        if not all(0.1 < s < 10 for s in spacing):
            results['valid_spacing'] = False
            results['pass'] = False
        else:
            results['valid_spacing'] = True
            
        if mask_path.exists():
            mask_nii = nib.load(str(mask_path))
            mask_data = mask_nii.get_fdata()
            
            if img_data.shape != mask_data.shape:
                results['shape_match'] = False
                results['pass'] = False
            else:
                results['shape_match'] = True
                
            unique_labels = np.unique(mask_data)
            if not np.all(np.isin(unique_labels, [0, 1, 2, 3, 4, 5, 6])):
                results['valid_labels'] = False
                results['pass'] = False
            else:
                results['valid_labels'] = True
                
            if len(unique_labels) == 1 and unique_labels[0] == 0:
                results['mask_not_empty'] = False
                results['pass'] = False
            else:
                results['mask_not_empty'] = True
                
            voxel_volume_ml = np.prod(spacing) / 1000.0
            
            liver_vol = np.sum(mask_data == 1) * voxel_volume_ml
            if 1 in unique_labels and not (500 < liver_vol < 5000):
                results['liver_vol_ok'] = False
                # results['pass'] = False # Optional depending on strictness
            else:
                results['liver_vol_ok'] = True
                
    except Exception as e:
        results['pass'] = False
        results['error'] = str(e)
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect corrupted cases.")
    parser.add_argument('--input-dir', type=Path, required=True, help="Input directory")
    parser.add_argument('--output-report', type=Path, required=True, help="Path to output CSV report")
    
    args = parser.parse_args()
    
    results = []
    for case_dir in args.input_dir.iterdir():
        if case_dir.is_dir():
            res = check_case(case_dir)
            results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv(args.output_report, index=False)
    logging.info(f"Report saved to {args.output_report}")
    
    failed = df[df['pass'] == False]
    if len(failed) > 0:
        logging.warning(f"Found {len(failed)} failed cases.")

if __name__ == "__main__":
    main()
