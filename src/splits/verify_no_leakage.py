"""
Module to verify that there is no data leakage between train/val/test splits.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_splits(splits_json: Path, manifest_path: Path | None = None) -> bool:
    """
    Verifies that splits have no data leakage.
    Returns True if passed, False otherwise.
    """
    with open(splits_json, 'r') as f:
        splits = json.load(f)
        
    test_set = set(splits.get("test", []))
    folds = splits.get("folds", {})
    
    passed = True
    
    # Check 1: No case in both train and val within any fold
    # Check 2: No case in both train/val and test
    # Check 4: No duplicate case IDs within any split
    all_cases_in_splits = set(test_set)
    for fold_idx, fold_data in folds.items():
        train_list = fold_data.get("train", [])
        val_list = fold_data.get("val", [])
        
        # Check for duplicates in list
        if len(train_list) != len(set(train_list)):
            logger.error(f"Duplicate cases found in Fold {fold_idx} train set")
            passed = False
        if len(val_list) != len(set(val_list)):
            logger.error(f"Duplicate cases found in Fold {fold_idx} val set")
            passed = False
            
        train_set = set(train_list)
        val_set = set(val_list)
        
        # Train vs Val leakage
        train_val_intersection = train_set.intersection(val_set)
        if train_val_intersection:
            logger.error(f"Leakage in Fold {fold_idx}: {len(train_val_intersection)} cases in both train and val.")
            passed = False
            
        # Train vs Test leakage
        train_test_intersection = train_set.intersection(test_set)
        if train_test_intersection:
            logger.error(f"Leakage in Fold {fold_idx}: {len(train_test_intersection)} cases in both train and test.")
            passed = False
            
        # Val vs Test leakage
        val_test_intersection = val_set.intersection(test_set)
        if val_test_intersection:
            logger.error(f"Leakage in Fold {fold_idx}: {len(val_test_intersection)} cases in both val and test.")
            passed = False
            
        all_cases_in_splits.update(train_set)
        all_cases_in_splits.update(val_set)
        
    # Check 3: All cases from manifest accounted for
    if manifest_path and manifest_path.exists():
        df = pd.read_csv(manifest_path)
        if 'global_id' not in df.columns:
            df['global_id'] = df.apply(lambda row: f"{row['dataset']}_{row['case_id']}", axis=1)
        
        manifest_cases = set(df['global_id'].tolist())
        
        missing_in_splits = manifest_cases - all_cases_in_splits
        if missing_in_splits:
            logger.error(f"Cases in manifest but missing from splits: {len(missing_in_splits)}")
            passed = False
            
        missing_in_manifest = all_cases_in_splits - manifest_cases
        if missing_in_manifest:
            logger.error(f"Cases in splits but missing from manifest: {len(missing_in_manifest)}")
            passed = False
            
    if passed:
        logger.info("PASS: No data leakage detected.")
        print("PASS")
    else:
        logger.error("FAIL: Data leakage or inconsistencies detected.")
        print("FAIL")
        
    return passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset splits for leakage")
    parser.add_argument("--splits-json", type=str, required=True, help="Path to splits JSON")
    parser.add_argument("--manifest", type=str, help="Path to manifest CSV (optional)")
    
    args = parser.parse_args()
    verify_splits(Path(args.splits_json), Path(args.manifest) if args.manifest else None)
