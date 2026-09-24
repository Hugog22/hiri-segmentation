"""
Module to convert datasets to a unified NIfTI format.
Handles DICOM -> NIfTI conversion and standardizes existing NIfTI datasets.
"""
import argparse
import logging
import shutil
import csv
from pathlib import Path
from typing import Dict, Any

import SimpleITK as sitk
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def convert_dicom_to_nifti(dicom_dir: Path, output_path: Path) -> None:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_path))


def process_dataset(dataset_name: str, raw_dir: Path, output_dir: Path) -> list:
    manifest = []
    dataset_out_dir = output_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name.lower() == 'chaos':
        # Assuming CHAOS has DICOM folders
        for case_dir in raw_dir.iterdir():
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            case_out = dataset_out_dir / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            out_img = case_out / 'image.nii.gz'
            dicom_dir = case_dir / 'DICOM_anon' # example
            if dicom_dir.exists():
                convert_dicom_to_nifti(dicom_dir, out_img)
                manifest.append({'case_id': case_id, 'dataset': dataset_name, 'original_path': str(dicom_dir), 'processed_path': str(out_img), 'contrast_phase': 'unknown'})
    elif dataset_name.lower() == 'totalsegmentator':
        # TotalSegmentator stores per-structure binary masks in segmentations/ folder.
        # Structure names for our target organs:
        TOTALSEG_ORGAN_FILES = {
            'liver': 5,          # unified label for liver
            'kidney_right': 2,   # unified label for right kidney
            'kidney_left': 3,    # unified label for left kidney
        }
        for case_dir in raw_dir.iterdir():
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            case_out = dataset_out_dir / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            img_src = case_dir / 'ct.nii.gz'
            img_dst = case_out / 'image.nii.gz'
            if not img_src.exists():
                logging.warning(f"No ct.nii.gz in {case_dir}, skipping")
                continue
            shutil.copy2(img_src, img_dst)
            # Combine per-structure binary masks into a single multi-label mask
            seg_dir = case_dir / 'segmentations'
            if seg_dir.exists():
                ref_img = sitk.ReadImage(str(img_src))
                combined = sitk.Image(ref_img.GetSize(), sitk.sitkUInt8)
                combined.CopyInformation(ref_img)
                combined_arr = sitk.GetArrayFromImage(combined)
                for organ_file, ts_label in TOTALSEG_ORGAN_FILES.items():
                    organ_path = seg_dir / f'{organ_file}.nii.gz'
                    if organ_path.exists():
                        organ_img = sitk.ReadImage(str(organ_path))
                        organ_arr = sitk.GetArrayFromImage(organ_img)
                        # Binary mask: set voxels to TotalSeg label ID
                        combined_arr[organ_arr > 0] = ts_label
                    else:
                        logging.warning(f"Missing {organ_file}.nii.gz for {case_id}")
                result = sitk.GetImageFromArray(combined_arr)
                result.CopyInformation(ref_img)
                sitk.WriteImage(result, str(case_out / 'label.nii.gz'))
            else:
                logging.warning(f"No segmentations/ directory for {case_id}")
            manifest.append({
                'case_id': case_id, 'dataset': dataset_name,
                'original_path': str(img_src),
                'processed_path': str(img_dst),
                'contrast_phase': 'unknown'
            })
    elif dataset_name.lower() == 'lits':
        for img_file in raw_dir.glob('volume-*.nii*'):
            case_id = img_file.name.split('-')[1].split('.')[0]
            case_out = dataset_out_dir / f"case_{case_id}"
            case_out.mkdir(parents=True, exist_ok=True)
            img_dst = case_out / 'image.nii.gz'
            shutil.copy2(img_file, img_dst)
            label_src = raw_dir / f"segmentation-{case_id}.nii"
            if label_src.exists():
                shutil.copy2(label_src, case_out / 'label.nii.gz')
            manifest.append({'case_id': f"case_{case_id}", 'dataset': dataset_name, 'original_path': str(img_file), 'processed_path': str(img_dst), 'contrast_phase': 'unknown'})
    elif dataset_name.lower() == 'kits23':
        for case_dir in raw_dir.glob('case_*'):
            case_id = case_dir.name
            case_out = dataset_out_dir / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            img_src = case_dir / 'imaging.nii.gz'
            label_src = case_dir / 'segmentation.nii.gz'
            if img_src.exists():
                img_dst = case_out / 'image.nii.gz'
                shutil.copy2(img_src, img_dst)
                if label_src.exists():
                    shutil.copy2(label_src, case_out / 'label.nii.gz')
                manifest.append({'case_id': case_id, 'dataset': dataset_name, 'original_path': str(img_src), 'processed_path': str(img_dst), 'contrast_phase': 'unknown'})
    elif dataset_name.lower() == 'amos':
        img_dir = raw_dir / 'imagesTr'
        label_dir = raw_dir / 'labelsTr'
        if img_dir.exists():
            for img_file in img_dir.glob('amos_*.nii.gz'):
                case_id = img_file.name.replace('.nii.gz', '')
                case_out = dataset_out_dir / case_id
                case_out.mkdir(parents=True, exist_ok=True)
                img_dst = case_out / 'image.nii.gz'
                shutil.copy2(img_file, img_dst)
                label_src = label_dir / f"{case_id}.nii.gz"
                if label_src.exists():
                    shutil.copy2(label_src, case_out / 'label.nii.gz')
                manifest.append({'case_id': case_id, 'dataset': dataset_name, 'original_path': str(img_file), 'processed_path': str(img_dst), 'contrast_phase': 'unknown'})
    elif dataset_name.lower() == 'btcv':
        img_dir = raw_dir / 'img'
        label_dir = raw_dir / 'label'
        if img_dir.exists():
            for img_file in img_dir.glob('img*.nii.gz'):
                case_id = img_file.name.replace('.nii.gz', '')
                case_out = dataset_out_dir / case_id
                case_out.mkdir(parents=True, exist_ok=True)
                img_dst = case_out / 'image.nii.gz'
                shutil.copy2(img_file, img_dst)
                label_num = case_id.replace('img', '')
                label_src = label_dir / f"label{label_num}.nii.gz"
                if label_src.exists():
                    shutil.copy2(label_src, case_out / 'label.nii.gz')
                manifest.append({'case_id': case_id, 'dataset': dataset_name, 'original_path': str(img_file), 'processed_path': str(img_dst), 'contrast_phase': 'unknown'})
    
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Convert datasets to a unified NIfTI format.")
    parser.add_argument('--dataset', required=True, help="Dataset name (e.g., TotalSegmentator, LiTS, KiTS23, AMOS, BTCV, CHAOS)")
    parser.add_argument('--raw-dir', type=Path, required=True, help="Directory containing raw dataset")
    parser.add_argument('--output-dir', type=Path, required=True, help="Output directory for processed NIfTIs")
    parser.add_argument('--config', type=Path, required=True, help="Path to base.yaml config")
    
    args = parser.parse_args()
    
    logging.info(f"Processing dataset {args.dataset} from {args.raw_dir}")
    manifest = process_dataset(args.dataset, args.raw_dir, args.output_dir)
    
    manifest_path = args.output_dir / f"{args.dataset}_manifest.csv"
    if manifest:
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['case_id', 'dataset', 'original_path', 'processed_path', 'contrast_phase'])
            writer.writeheader()
            writer.writerows(manifest)
        logging.info(f"Saved manifest to {manifest_path}")
    else:
        logging.warning("No files processed.")

if __name__ == "__main__":
    main()
