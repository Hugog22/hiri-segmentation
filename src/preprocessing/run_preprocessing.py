"""
Master orchestration script to run the full preprocessing pipeline.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: list) -> None:
    logging.info(f"Running: {' '.join([str(c) for c in cmd])}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join([str(c) for c in cmd])}")


def main():
    parser = argparse.ArgumentParser(description="Master preprocessing script.")
    parser.add_argument('--config', type=Path, default=Path('configs/base.yaml'), help="Path to config yaml")
    parser.add_argument('--steps', nargs='+', default=['convert', 'unify', 'verify', 'laterality', 'corrupt', 'resample'], help="Steps to run")
    parser.add_argument('--datasets', nargs='+', default=['TotalSegmentator', 'LiTS', 'KiTS23', 'AMOS', 'BTCV'], help="Datasets to process")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config.exists():
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
        
    project_root = Path.cwd()
    data_cfg = cfg.get('data', {})
    
    raw_data_dir = project_root / data_cfg.get('raw_dir', 'data/raw')
    processed_dir = project_root / data_cfg.get('processed_dir', 'data/processed')
    nnunet_raw = Path(os.environ.get('nnUNet_raw', project_root / 'nnUNet_raw'))
    
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    nnunet_raw.mkdir(parents=True, exist_ok=True)
    
    py_exec = sys.executable
    base_src_dir = Path(__file__).resolve().parent
    
    datasets_processed = []
    
    for dataset in args.datasets:
        ds_raw = raw_data_dir / dataset
        ds_proc = processed_dir / dataset
        
        if not ds_raw.exists() or not any(ds_raw.iterdir()):
            logging.warning(f"Dataset directory '{ds_raw}' is empty or does not exist. Skipping {dataset}.")
            continue
            
        logging.info(f"=== Processing Dataset: {dataset} ===")
        datasets_processed.append(dataset)
        
        if 'convert' in args.steps:
            run_command([py_exec, str(base_src_dir / 'convert_to_nifti.py'), '--dataset', dataset, '--raw-dir', str(ds_raw), '--output-dir', str(processed_dir), '--config', str(args.config)])
            
        if 'unify' in args.steps and ds_proc.exists():
            run_command([py_exec, str(base_src_dir / 'unify_labels.py'), '--dataset', dataset, '--input-dir', str(ds_proc), '--output-dir', str(ds_proc)])
            
        if 'verify' in args.steps and ds_proc.exists():
            run_command([py_exec, str(base_src_dir / 'verify_orientation.py'), '--input-dir', str(ds_proc), '--fix'])
            
        if 'laterality' in args.steps and dataset.lower() == 'kits23' and ds_proc.exists():
            run_command([py_exec, str(base_src_dir / 'resolve_kits_laterality.py'), '--input-dir', str(ds_proc), '--output-dir', str(ds_proc), '--report-path', str(ds_proc / 'laterality_report.csv')])
            
        if 'corrupt' in args.steps and ds_proc.exists():
            run_command([py_exec, str(base_src_dir / 'detect_corrupted.py'), '--input-dir', str(ds_proc), '--output-report', str(ds_proc / 'corrupted_report.csv')])
            
        if 'resample' in args.steps and ds_proc.exists():
            run_command([py_exec, str(base_src_dir / 'resample.py'), '--input-dir', str(ds_proc), '--output-dir', str(ds_proc / 'resampled')])

    if not datasets_processed:
        logging.warning("No datasets found in data/raw/. Please download datasets first via 'python scripts/download_datasets.py --download-public'.")
    else:
        logging.info(f"Successfully processed datasets: {datasets_processed}")

    # Aggregate all *_manifest.csv into a unified master manifest.csv
    import pandas as pd
    manifest_files = list(processed_dir.glob("*_manifest.csv"))
    if manifest_files:
        dfs = [pd.read_csv(f) for f in manifest_files]
        master_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["case_id", "dataset"])
        master_path = processed_dir / "manifest.csv"
        master_df.to_csv(master_path, index=False)
        logging.info(f"Generated master manifest at {master_path} with {len(master_df)} total cases.")

if __name__ == "__main__":
    main()
