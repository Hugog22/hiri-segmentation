"""
Module to resample images and masks to target spacing.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resample_scipy(input_path: Path, output_path: Path, target_spacing: List[float], is_mask: bool) -> None:
    nii = nib.load(str(input_path))
    data = np.asarray(nii.dataobj)
    orig_spacing = list(nii.header.get_zooms()[:3])
    zoom_factors = [orig_s / target_s for orig_s, target_s in zip(orig_spacing, target_spacing)]
    
    order = 0 if is_mask else 3
    resampled_data = zoom(data, zoom_factors, order=order)
    
    new_affine = np.copy(nii.affine)
    for i in range(3):
        new_affine[:3, i] = new_affine[:3, i] * (target_spacing[i] / orig_spacing[i])
        
    out_nii = nib.Nifti1Image(resampled_data.astype(data.dtype), new_affine, nii.header)
    nib.save(out_nii, str(output_path))


def resample_image(input_path: Path, output_path: Path, target_spacing: List[float], interpolation: str = 'bspline') -> None:
    try:
        image = sitk.ReadImage(str(input_path))
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        
        if interpolation == 'bspline':
            resampler.SetInterpolator(sitk.sitkBSpline)
        elif interpolation == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        else:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            
        resampled_image = resampler.Execute(image)
        sitk.WriteImage(resampled_image, str(output_path))
    except Exception as e:
        logger.warning(f"SimpleITK failed on {input_path.name} ({e}). Falling back to nibabel/scipy resampling.")
        _resample_scipy(input_path, output_path, target_spacing, is_mask=(interpolation == 'nearest'))


def resample_mask(input_path: Path, output_path: Path, target_spacing: List[float]) -> None:
    resample_image(input_path, output_path, target_spacing, interpolation='nearest')


def save_original_metadata(image_path: Path, json_path: Path) -> None:
    try:
        image = sitk.ReadImage(str(image_path))
        meta = {
            'spacing': list(image.GetSpacing()),
            'shape': list(image.GetSize()),
            'origin': list(image.GetOrigin()),
            'direction': list(image.GetDirection())
        }
    except Exception:
        nii = nib.load(str(image_path))
        meta = {
            'spacing': [float(s) for s in nii.header.get_zooms()[:3]],
            'shape': list(nii.shape),
            'origin': [float(x) for x in nii.affine[:3, 3]],
            'direction': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Resample images and masks.")
    parser.add_argument('--input-dir', type=Path, required=True, help="Input directory")
    parser.add_argument('--output-dir', type=Path, required=True, help="Output directory")
    parser.add_argument('--target-spacing', type=float, nargs=3, default=[1.0, 1.0, 1.5], help="Target spacing (x y z)")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for case_dir in args.input_dir.iterdir():
        if not case_dir.is_dir() or case_dir.name == 'resampled':
            continue
            
        img_path = case_dir / 'image.nii.gz'
        mask_path = case_dir / 'unified_label.nii.gz'
        if not mask_path.exists():
            mask_path = case_dir / 'label.nii.gz'
            
        if not img_path.exists() and not mask_path.exists():
            continue

        case_out_dir = args.output_dir / case_dir.name
        case_out_dir.mkdir(parents=True, exist_ok=True)
        
        img_out = case_out_dir / 'image.nii.gz'
        mask_out = case_out_dir / 'unified_label.nii.gz'
        
        already_done = True
        if img_path.exists() and not img_out.exists():
            already_done = False
        if mask_path.exists() and not mask_out.exists():
            already_done = False
            
        if already_done:
            continue

        if img_path.exists() and not img_out.exists():
            save_original_metadata(img_path, case_out_dir / 'original_metadata.json')
            resample_image(img_path, img_out, args.target_spacing)
            
        if mask_path.exists() and not mask_out.exists():
            resample_mask(mask_path, mask_out, args.target_spacing)
            
        logging.info(f"Resampled {case_dir.name}")

if __name__ == "__main__":
    main()
