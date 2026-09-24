"""
Module to create train, validation, and test splits for the HI&RI project.
Splits are stratified by dataset and optionally by contrast_phase.
Generates a JSON file containing the splits.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_splits(manifest_path: Path, output_path: Path, seed: int = 42, n_folds: int = 5) -> None:
    """
    Create dataset splits from a manifest CSV.
    Holds out 20% for test, and creates cross-validation folds on the remaining 80%.
    """
    logger.info(f"Loading manifest from {manifest_path}")
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    
    # Ensure necessary columns exist
    required_cols = ['case_id', 'dataset']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from manifest.")
            
    # Create global unique ID
    df['global_id'] = df.apply(lambda row: f"{row['dataset']}_{row['case_id']}", axis=1)
    
    # Stratification column
    if 'contrast_phase' in df.columns:
        df['stratify_col'] = df['dataset'] + "_" + df['contrast_phase'].fillna('unknown')
    else:
        df['stratify_col'] = df['dataset']
        
    # Remove cases where stratify_col has less than 2 instances if needed, or group them
    val_counts = df['stratify_col'].value_counts()
    rare_strats = val_counts[val_counts < 2].index
    df.loc[df['stratify_col'].isin(rare_strats), 'stratify_col'] = 'rare_group'
    
    # 1. Hold out 20% for TEST
    logger.info("Splitting 20% for test set...")
    train_val_df, test_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df['stratify_col']
    )
    
    test_cases = test_df['global_id'].tolist()
    
    # 2. Create 5-fold CV on train_val_df
    logger.info(f"Creating {n_folds}-fold CV on remaining 80%...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Reset index for SKF
    train_val_df = train_val_df.reset_index(drop=True)
    
    folds_dict = {}
    
    # Check stratification for folds
    fold_stratify = train_val_df['stratify_col']
    fold_val_counts = fold_stratify.value_counts()
    rare_fold_strats = fold_val_counts[fold_val_counts < n_folds].index
    train_val_df.loc[train_val_df['stratify_col'].isin(rare_fold_strats), 'stratify_col'] = 'rare_fold_group'

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['stratify_col'])):
        fold_train = train_val_df.iloc[train_idx]['global_id'].tolist()
        fold_val = train_val_df.iloc[val_idx]['global_id'].tolist()
        folds_dict[str(fold_idx)] = {
            "train": fold_train,
            "val": fold_val
        }
        
    # 3. Create structure
    splits_structure = {
        "seed": seed,
        "test": test_cases,
        "folds": folds_dict,
        "external_validation": {
            "exp_amos": {"train_datasets": ["totalsegmentator", "lits", "kits23", "btcv"], "eval_dataset": "amos22"},
            "exp_kits": {"train_datasets": ["totalsegmentator", "lits", "amos22", "btcv"], "eval_dataset": "kits23"}
        }
    }
    
    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits_structure, f, indent=4)
        
    logger.info(f"Splits saved to {output_path}")
    
    # Print statistics
    print("\n--- SPLIT STATISTICS ---")
    print(f"Total cases: {len(df)}")
    print(f"Test cases: {len(test_cases)}")
    for fold, data in folds_dict.items():
        print(f"Fold {fold}: {len(data['train'])} train, {len(data['val'])} val")
        
    print("\nDataset distribution in TEST set:")
    print(test_df['dataset'].value_counts().to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--output", type=str, default="splits.json", help="Path to output JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    create_splits(Path(args.manifest), Path(args.output), seed=args.seed)
