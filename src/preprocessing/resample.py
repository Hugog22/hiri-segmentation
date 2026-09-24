"""
Module to resample images and masks to target spacing.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List

import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def resample_image(input_path: Path, output_path: Path, target_spacing: List[float], interpolation: str = 'bspline') -> None:
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


def resample_mask(input_path: Path, output_path: Path, target_spacing: List[float]) -> None:
    resample_image(input_path, output_path, target_spacing, interpolation='nearest')


def save_original_metadata(image_path: Path, json_path: Path) -> None:
    image = sitk.ReadImage(str(image_path))
    meta = {
        'spacing': image.GetSpacing(),
        'shape': image.GetSize(),
        'origin': image.GetOrigin(),
        'direction': image.GetDirection()
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
        if not case_dir.is_dir():
            continue
            
        case_out_dir = args.output_dir / case_dir.name
        case_out_dir.mkdir(parents=True, exist_ok=True)
        
        img_path = case_dir / 'image.nii.gz'
        mask_path = case_dir / 'unified_label.nii.gz'
        
        if img_path.exists():
            save_original_metadata(img_path, case_out_dir / 'original_metadata.json')
            resample_image(img_path, case_out_dir / 'image.nii.gz', args.target_spacing)
            
        if mask_path.exists():
            resample_mask(mask_path, case_out_dir / 'unified_label.nii.gz', args.target_spacing)
            
        logging.info(f"Resampled {case_dir.name}")

if __name__ == "__main__":
    main()
