"""
Module to prepare the dataset in nnU-Net v2 format.
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Prepare nnU-Net v2 dataset.")
    parser.add_argument('--input-dir', type=Path, required=True, help="Input directory")
    parser.add_argument('--output-dir', type=Path, required=True, help="nnU-Net raw dataset directory")
    parser.add_argument('--dataset-id', type=int, required=True, help="Dataset ID (e.g. 100)")
    parser.add_argument('--splits-json', type=Path, help="Optional splits definition")
    
    args = parser.parse_args()
    
    ds_name = f"Dataset{args.dataset_id:03d}_HIRI"
    base_dir = args.output_dir / ds_name
    
    img_tr = base_dir / 'imagesTr'
    lbl_tr = base_dir / 'labelsTr'
    img_ts = base_dir / 'imagesTs'
    lbl_ts = base_dir / 'labelsTs'
    
    for d in [img_tr, lbl_tr, img_ts, lbl_ts]:
        d.mkdir(parents=True, exist_ok=True)
        
    num_training = 0
    
    # Simplistic copy (assume all go to Tr for now unless split defines otherwise)
    for case_dir in args.input_dir.iterdir():
        if not case_dir.is_dir():
            continue
            
        case_id = f"HIRI_{case_dir.name}"
        img_src = case_dir / 'image.nii.gz'
        lbl_src = case_dir / 'unified_label.nii.gz'
        
        if img_src.exists() and lbl_src.exists():
            shutil.copy2(img_src, img_tr / f"{case_id}_0000.nii.gz")
            shutil.copy2(lbl_src, lbl_tr / f"{case_id}.nii.gz")
            num_training += 1
            logging.info(f"Copied {case_id}")
            
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "liver": 1,
            "right_kidney": 2,
            "left_kidney": 3,
            "hepatic_tumor": 4,
            "renal_tumor": 5,
            "renal_cyst": 6
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    
    with open(base_dir / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=4)
        
    logging.info(f"Dataset prepared at {base_dir}")

if __name__ == "__main__":
    main()
