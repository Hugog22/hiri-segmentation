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


def has_raw_data(dataset: str, raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return False
    name = dataset.lower()
    if name == 'totalsegmentator':
        return any(raw_dir.glob('s*')) or any((d / 'ct.nii.gz').exists() for d in raw_dir.iterdir() if d.is_dir())
    elif name == 'kits23':
        return any(raw_dir.glob('case_*'))
    elif name == 'lits':
        return any(raw_dir.glob('volume-*.nii*'))
    elif name in ['amos', 'amos22']:
        return (raw_dir / 'imagesTr').exists() and any((raw_dir / 'imagesTr').glob('*.nii*'))
    elif name == 'btcv':
        return (raw_dir / 'img').exists() and any((raw_dir / 'img').glob('*.nii*'))
    elif name == 'chaos':
        return any(d.is_dir() for d in raw_dir.iterdir())
    return any(raw_dir.iterdir())


def has_processed_data(proc_dir: Path) -> bool:
    if not proc_dir.exists():
        return False
    return any(
        d.is_dir() and (d / 'image.nii.gz').exists() 
        for d in proc_dir.iterdir() 
        if d.name != 'resampled'
    )


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
        
        raw_available = has_raw_data(dataset, ds_raw)
        proc_available = has_processed_data(ds_proc)

        if not raw_available and not proc_available:
            logging.warning(f"Dataset '{dataset}' has no raw data in '{ds_raw}' and no processed data in '{ds_proc}'. Skipping.")
            continue
            
        logging.info(f"=== Processing Dataset: {dataset} ===")
        
        if 'convert' in args.steps:
            if raw_available:
                run_command([py_exec, str(base_src_dir / 'convert_to_nifti.py'), '--dataset', dataset, '--raw-dir', str(ds_raw), '--output-dir', str(processed_dir), '--config', str(args.config)])
            else:
                logging.info(f"Skipping conversion for {dataset} (no raw volumes found, using existing processed data).")
            
        if not has_processed_data(ds_proc):
            logging.warning(f"No valid converted cases found for {dataset} in {ds_proc}. Skipping remaining steps for {dataset}.")
            continue

        datasets_processed.append(dataset)

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
        logging.warning("No datasets found in data/raw/ or data/processed/. Please download datasets first via 'python scripts/download_datasets.py --download-public'.")
    else:
        logging.info(f"Successfully processed datasets: {datasets_processed}")

    # Aggregate all *_manifest.csv into a unified master manifest.csv
    import pandas as pd
    manifest_files = [f for f in processed_dir.glob("*_manifest.csv") if f.stat().st_size > 0]
    master_path = processed_dir / "manifest.csv"
    if manifest_files:
        dfs = []
        for f in manifest_files:
            try:
                df_m = pd.read_csv(f)
                if not df_m.empty:
                    dfs.append(df_m)
            except Exception as e:
                logging.warning(f"Could not read manifest {f}: {e}")
        if dfs:
            master_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["case_id", "dataset"])
            master_df.to_csv(master_path, index=False)
            logging.info(f"Generated master manifest at {master_path} with {len(master_df)} total cases.")
    
    # Fallback if no individual manifests or manifest.csv does not exist yet
    if not master_path.exists():
        rows = []
        for ds_dir in processed_dir.iterdir():
            if not ds_dir.is_dir() or ds_dir.name.startswith('.'):
                continue
            for case_dir in ds_dir.iterdir():
                if case_dir.is_dir() and case_dir.name != 'resampled' and (case_dir / 'image.nii.gz').exists():
                    rows.append({
                        'case_id': case_dir.name,
                        'dataset': ds_dir.name,
                        'original_path': str(case_dir / 'image.nii.gz'),
                        'processed_path': str(case_dir / 'image.nii.gz'),
                        'contrast_phase': 'unknown'
                    })
        if rows:
            master_df = pd.DataFrame(rows).drop_duplicates(subset=["case_id", "dataset"])
            master_df.to_csv(master_path, index=False)
            logging.info(f"Generated fallback master manifest at {master_path} with {len(master_df)} cases.")

if __name__ == "__main__":
    main()
