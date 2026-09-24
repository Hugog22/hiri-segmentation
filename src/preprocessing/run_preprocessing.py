"""
Master orchestration script to run the full preprocessing pipeline.
"""
import argparse
import logging
import subprocess
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_command(cmd: list) -> None:
    logging.info(f"Running: {' '.join([str(c) for c in cmd])}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join([str(c) for c in cmd])}")

def main():
    parser = argparse.ArgumentParser(description="Master preprocessing script.")
    parser.add_argument('--config', type=Path, default=Path('../../configs/base.yaml'), help="Path to config yaml")
    parser.add_argument('--steps', nargs='+', default=['convert', 'unify', 'verify', 'laterality', 'corrupt', 'resample', 'nnunet'], help="Steps to run")
    parser.add_argument('--datasets', nargs='+', default=['TotalSegmentator', 'LiTS', 'KiTS23', 'AMOS', 'BTCV'], help="Datasets to process")
    
    args = parser.parse_args()
    
    base_src_dir = Path(__file__).parent
    
    # Using mock paths for demonstration, in reality these would come from base.yaml
    raw_data_dir = Path('/Users/hugo/Documents/PROYECTOS/HI&RI/data/raw')
    processed_dir = Path('/Users/hugo/Documents/PROYECTOS/HI&RI/data/processed')
    nnunet_raw = Path('/Users/hugo/Documents/PROYECTOS/HI&RI/data/nnUNet_raw')
    
    for dataset in args.datasets:
        ds_raw = raw_data_dir / dataset
        ds_proc = processed_dir / dataset
        
        if 'convert' in args.steps:
            run_command(['python', str(base_src_dir / 'convert_to_nifti.py'), '--dataset', dataset, '--raw-dir', str(ds_raw), '--output-dir', str(processed_dir), '--config', str(args.config)])
            
        if 'unify' in args.steps:
            run_command(['python', str(base_src_dir / 'unify_labels.py'), '--dataset', dataset, '--input-dir', str(ds_proc), '--output-dir', str(ds_proc)])
            
        if 'verify' in args.steps:
            run_command(['python', str(base_src_dir / 'verify_orientation.py'), '--input-dir', str(ds_proc), '--fix'])
            
        if 'laterality' in args.steps and dataset == 'KiTS23':
            run_command(['python', str(base_src_dir / 'resolve_kits_laterality.py'), '--input-dir', str(ds_proc), '--output-dir', str(ds_proc), '--report-path', str(ds_proc / 'laterality_report.csv')])
            
        if 'corrupt' in args.steps:
            run_command(['python', str(base_src_dir / 'detect_corrupted.py'), '--input-dir', str(ds_proc), '--output-report', str(ds_proc / 'corrupted_report.csv')])
            
        if 'resample' in args.steps:
            run_command(['python', str(base_src_dir / 'resample.py'), '--input-dir', str(ds_proc), '--output-dir', str(ds_proc / 'resampled')])
            
    if 'nnunet' in args.steps:
        # Assuming we bundle everything from processed into nnUNet format
        # This part might need careful adaptation to handle multiple datasets into one dataset ID
        run_command(['python', str(base_src_dir / 'prepare_nnunet_dataset.py'), '--input-dir', str(processed_dir / 'TotalSegmentator' / 'resampled'), '--output-dir', str(nnunet_raw), '--dataset-id', '100'])

    logging.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
