"""
Module to prepare the dataset in nnU-Net v2 format.
Creates DatasetXXX_HIRI inside nnUNet_raw, creates dataset.json,
and generates splits_final.json compatible with nnU-Net v2 cross-validation.
"""
import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def link_or_copy(src: Path, dst: Path) -> None:
    """Create symlink to save disk space, fallback to copy if symlinks not supported."""
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def find_case_files(input_dir: Path, global_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locates image.nii.gz and unified_label.nii.gz for a given global case ID.
    Example global_id: 'TotalSegmentator_s0001'
    """
    parts = global_id.split("_", 1)
    dataset = parts[0]
    case_subid = parts[1] if len(parts) > 1 else parts[0]

    candidate_dirs = [
        input_dir / dataset / case_subid,
        input_dir / dataset / global_id,
        input_dir / case_subid,
        input_dir / global_id,
    ]

    # Also case-insensitive check for dataset folder name
    for d in input_dir.iterdir():
        if d.is_dir() and d.name.lower() == dataset.lower():
            candidate_dirs.append(d / case_subid)
            candidate_dirs.append(d / global_id)

    for cand in candidate_dirs:
        if cand.exists() and cand.is_dir():
            img = cand / 'image.nii.gz'
            lbl = cand / 'unified_label.nii.gz'
            if not lbl.exists():
                lbl = cand / 'label.nii.gz'
            if img.exists() and lbl.exists():
                return img, lbl

    return None, None


def discover_all_cases(input_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Discovers all available processed cases in input_dir."""
    cases = {}
    for item in input_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.') or item.name == 'resampled':
            continue
        # Direct case folder
        if (item / 'image.nii.gz').exists():
            lbl = item / 'unified_label.nii.gz'
            if not lbl.exists():
                lbl = item / 'label.nii.gz'
            if lbl.exists():
                cases[item.name] = (item / 'image.nii.gz', lbl)
        else:
            # Dataset folder containing case folders
            for sub in item.iterdir():
                if sub.is_dir() and sub.name != 'resampled':
                    img = sub / 'image.nii.gz'
                    lbl = sub / 'unified_label.nii.gz'
                    if not lbl.exists():
                        lbl = sub / 'label.nii.gz'
                    if img.exists() and lbl.exists():
                        cases[f"{item.name}_{sub.name}"] = (img, lbl)
    return cases


def main():
    parser = argparse.ArgumentParser(description="Prepare nnU-Net v2 dataset.")
    parser.add_argument('--input-dir', type=Path, required=True, help="Input processed directory")
    parser.add_argument('--output-dir', type=Path, required=True, help="nnU-Net raw dataset directory")
    parser.add_argument('--dataset-id', type=int, default=100, help="Dataset ID (default: 100)")
    parser.add_argument('--splits-json', type=Path, help="Path to splits.json")

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
    all_cases = discover_all_cases(args.input_dir)
    logger.info(f"Discovered {len(all_cases)} total processed cases across datasets.")

    splits_data = None
    if args.splits_json and args.splits_json.exists():
        with open(args.splits_json, 'r') as f:
            splits_data = json.load(f)

    if splits_data:
        test_ids = set(splits_data.get('test', []))
        folds_info = splits_data.get('folds', {})
        train_ids = set()
        for fold in folds_info.values():
            train_ids.update(fold.get('train', []))
            train_ids.update(fold.get('val', []))

        # Copy/link train cases
        for cid in train_ids:
            img, lbl = find_case_files(args.input_dir, cid)
            if img and lbl:
                case_id = f"HIRI_{cid}"
                link_or_copy(img, img_tr / f"{case_id}_0000.nii.gz")
                link_or_copy(lbl, lbl_tr / f"{case_id}.nii.gz")
                num_training += 1
            else:
                logger.warning(f"Train case {cid} not found on disk, skipping.")

        # Copy/link test cases
        for cid in test_ids:
            img, lbl = find_case_files(args.input_dir, cid)
            if img and lbl:
                case_id = f"HIRI_{cid}"
                link_or_copy(img, img_ts / f"{case_id}_0000.nii.gz")
                link_or_copy(lbl, lbl_ts / f"{case_id}.nii.gz")

        # Create nnU-Net v2 splits_final.json
        nnunet_folds = []
        for fold_idx in sorted(folds_info.keys(), key=lambda x: int(x)):
            f_data = folds_info[fold_idx]
            tr_list = [f"HIRI_{c}" for c in f_data.get('train', []) if (img_tr / f"HIRI_{c}_0000.nii.gz").exists()]
            val_list = [f"HIRI_{c}" for c in f_data.get('val', []) if (img_tr / f"HIRI_{c}_0000.nii.gz").exists()]
            nnunet_folds.append({"train": tr_list, "val": val_list})

        # Save to base_dir
        with open(base_dir / "splits_final.json", "w") as f:
            json.dump(nnunet_folds, f, indent=4)
        logger.info(f"Saved nnU-Net splits_final.json to {base_dir / 'splits_final.json'}")

        # Also save directly to nnUNet_preprocessed if env variable or path is known
        preprocessed_base = Path(os.environ.get('nnUNet_preprocessed', Path.cwd() / 'nnUNet_preprocessed'))
        prep_ds_dir = preprocessed_base / ds_name
        prep_ds_dir.mkdir(parents=True, exist_ok=True)
        with open(prep_ds_dir / "splits_final.json", "w") as f:
            json.dump(nnunet_folds, f, indent=4)
        logger.info(f"Installed splits_final.json to {prep_ds_dir / 'splits_final.json'}")

    else:
        # No splits provided: link all discovered cases to imagesTr
        for cid, (img, lbl) in all_cases.items():
            case_id = f"HIRI_{cid}"
            link_or_copy(img, img_tr / f"{case_id}_0000.nii.gz")
            link_or_copy(lbl, lbl_tr / f"{case_id}.nii.gz")
            num_training += 1

    # Check foreground labels in dataset
    has_tumors = False
    sample_lbls = list(lbl_tr.glob("*.nii.gz"))[:10]
    for sp in sample_lbls:
        try:
            arr = np.asarray(nib.load(str(sp)).dataobj)
            if np.any(arr >= 4):
                has_tumors = True
                break
        except Exception:
            pass

    labels = {
        "background": 0,
        "liver": 1,
        "right_kidney": 2,
        "left_kidney": 3
    }
    if has_tumors:
        labels["hepatic_tumor"] = 4
        labels["renal_tumor"] = 5
        labels["renal_cyst"] = 6

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }

    with open(base_dir / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=4)

    logger.info(f"Dataset100_HIRI prepared at {base_dir} with {num_training} training cases.")


if __name__ == "__main__":
    main()
