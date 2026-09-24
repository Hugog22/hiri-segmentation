"""
Module to verify and fix the orientation of medical images.
Ensures all images are in RAS orientation.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import nibabel as nib
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_orientation(nifti_path: Path) -> str:
    nii = nib.load(str(nifti_path))
    aff = nii.affine
    ornt = nib.io_orientation(aff)
    ornt_str = nib.orientations.ornt2axcodes(ornt)
    return "".join(ornt_str)

def reorient_to_ras(nifti_path: Path, output_path: Path) -> None:
    nii = nib.load(str(nifti_path))
    nii_ras = nib.as_closest_canonical(nii)
    nib.save(nii_ras, str(output_path))

def verify_image_mask_consistency(image_path: Path, mask_path: Path) -> Dict[str, Any]:
    img = nib.load(str(image_path))
    mask = nib.load(str(mask_path))
    
    shape_match = img.shape == mask.shape
    import numpy as np
    affine_match = np.allclose(img.affine, mask.affine, atol=1e-3)
    
    return {
        'shape_match': shape_match,
        'affine_match': affine_match,
        'img_orientation': get_orientation(image_path),
        'mask_orientation': get_orientation(mask_path)
    }

def batch_verify(input_dir: Path, fix: bool = False) -> pd.DataFrame:
    results = []
    for case_dir in input_dir.iterdir():
        if not case_dir.is_dir():
            continue
            
        img_path = case_dir / 'image.nii.gz'
        mask_path = case_dir / 'unified_label.nii.gz'
        if not mask_path.exists():
            mask_path = case_dir / 'label.nii.gz'
            
        if not img_path.exists() or not mask_path.exists():
            continue
            
        img_ornt = get_orientation(img_path)
        mask_ornt = get_orientation(mask_path)
        
        consistency = verify_image_mask_consistency(img_path, mask_path)
        consistency['case_id'] = case_dir.name
        
        if fix:
            if img_ornt != 'RAS':
                reorient_to_ras(img_path, img_path)
                consistency['img_orientation'] = 'RAS (fixed)'
            if mask_ornt != 'RAS':
                reorient_to_ras(mask_path, mask_path)
                consistency['mask_orientation'] = 'RAS (fixed)'
                
        results.append(consistency)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Verify and fix NIfTI orientation to RAS.")
    parser.add_argument('--input-dir', type=Path, required=True, help="Input directory")
    parser.add_argument('--fix', action='store_true', help="Auto-fix to RAS")
    parser.add_argument('--report-only', action='store_true', help="Only report, don't fix")
    
    args = parser.parse_args()
    
    df = batch_verify(args.input_dir, fix=args.fix and not args.report_only)
    logging.info(f"\n{df.to_string()}")

if __name__ == "__main__":
    main()
