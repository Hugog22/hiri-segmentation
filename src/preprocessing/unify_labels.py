"""
Module to convert dataset-specific labels to the unified label map.
Unified Labels: 0=background, 1=liver, 2=right_kidney, 3=left_kidney, 4=hepatic_tumor, 5=renal_tumor, 6=renal_cyst
"""
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def unify_labels_for_dataset(dataset_name: str, input_dir: Path, output_dir: Path, merge_tumors: bool) -> None:
    for case_dir in input_dir.iterdir():
        if not case_dir.is_dir():
            continue
        
        label_path = case_dir / 'label.nii.gz'
        if not label_path.exists():
            continue
        
        nii = nib.load(str(label_path))
        data = nii.get_fdata()
        unified_data = np.zeros_like(data)
        
        if dataset_name.lower() == 'totalsegmentator':
            unified_data[data == 5] = 1
            unified_data[data == 2] = 2
            unified_data[data == 3] = 3
        elif dataset_name.lower() == 'lits':
            unified_data[data == 1] = 1
            if merge_tumors:
                unified_data[data == 2] = 1
            else:
                unified_data[data == 2] = 4
        elif dataset_name.lower() == 'kits23':
            # 1->needs laterality (assume another script resolves 1->2/3)
            # if 1 exists, we just copy it and let resolve script handle it, or we leave it.
            unified_data[data == 1] = 1 
            unified_data[data == 2] = 5
            unified_data[data == 3] = 6
        elif dataset_name.lower() in ['amos', 'btcv']:
            unified_data[data == 6] = 1
            unified_data[data == 2] = 2
            unified_data[data == 3] = 3
        else:
            unified_data = data
            
        case_out_dir = output_dir / case_dir.name
        case_out_dir.mkdir(parents=True, exist_ok=True)
        
        out_nii = nib.Nifti1Image(unified_data.astype(np.uint8), nii.affine, nii.header)
        out_path = case_out_dir / 'unified_label.nii.gz'
        nib.save(out_nii, str(out_path))
        
        unique, counts = np.unique(unified_data, return_counts=True)
        stats = dict(zip(unique, counts))
        logging.info(f"Processed {case_dir.name}: {stats}")

def main():
    parser = argparse.ArgumentParser(description="Convert labels to unified label map.")
    parser.add_argument('--dataset', required=True, help="Dataset name")
    parser.add_argument('--input-dir', type=Path, required=True, help="Input directory")
    parser.add_argument('--output-dir', type=Path, required=True, help="Output directory")
    parser.add_argument('--merge-tumors', action='store_true', help="Merge tumors into main organs")
    
    args = parser.parse_args()
    
    unify_labels_for_dataset(args.dataset, args.input_dir, args.output_dir, args.merge_tumors)

if __name__ == "__main__":
    main()
