"""
Master experiment orchestrator for HI&RI segmentation.
Manages nnU-Net and MONAI models training and evaluation.
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS = {
    'M1_nnunet_3d_fullres': {'framework': 'nnunet', 'config': '3d_fullres', 'planner': 'nnUNetPlanner'},
    'M2_nnunet_cascade': {'framework': 'nnunet', 'config': '3d_cascade_fullres', 'planner': 'nnUNetPlanner'},
    'M3_nnunet_resenc_l': {'framework': 'nnunet', 'config': '3d_fullres', 'planner': 'nnUNetPlannerResEncL'},
    'M4_nnunet_resenc_xl': {'framework': 'nnunet', 'config': '3d_fullres', 'planner': 'nnUNetPlannerResEncXL'},
    'M5_swin_unetr_base': {'framework': 'monai', 'model': 'swin_unetr_base', 'pretrained': True},
    'M6_swin_unetr_large': {'framework': 'monai', 'model': 'swin_unetr_large', 'pretrained': False},
    'M7_unetr': {'framework': 'monai', 'model': 'unetr', 'pretrained': False},
}

def load_status(status_file: Path) -> Dict[str, Any]:
    if status_file.exists():
        with open(status_file, "r") as f:
            return json.load(f)
    return {}

def save_status(status_file: Path, status: Dict[str, Any]) -> None:
    with open(status_file, "w") as f:
        json.dump(status, f, indent=4)

def run_single_experiment(exp_name: str, exp_config: Dict[str, Any], folds: List[int], config_path: str, output_dir: Path) -> None:
    """Run all folds for one experiment."""
    logger.info(f"Running experiment {exp_name} on folds {folds}")
    framework = exp_config.get("framework")
    
    if framework == "nnunet":
        dataset_id = 100
        config = exp_config.get("config", "3d_fullres")
        planner = exp_config.get("planner", "nnUNetPlanner")
        for fold in folds:
            cmd = ["python", "-m", "src.training.run_nnunet", "train", 
                   "--dataset-id", str(dataset_id), "--config", config, "--fold", str(fold), "--planner", planner]
            subprocess.run(cmd, check=True)
    elif framework == "monai":
        model = exp_config.get("model")
        for fold in folds:
            cmd = ["python", "-m", "src.training.monai_trainer", "--model", model, "--fold", str(fold), "--config", config_path]
            subprocess.run(cmd, check=True)
    else:
        logger.error(f"Unknown framework: {framework}")
        sys.exit(1)

def run_all_experiments(experiments: Dict[str, Any], folds: List[int], config_path: str, output_dir: Path) -> None:
    """Run all experiments sequentially."""
    status_file = output_dir / "experiment_status.json"
    status = load_status(status_file)
    
    for exp_name, exp_config in experiments.items():
        if status.get(exp_name) == "completed":
            logger.info(f"Skipping {exp_name}, already completed.")
            continue
            
        status[exp_name] = "running"
        save_status(status_file, status)
        try:
            run_single_experiment(exp_name, exp_config, folds, config_path, output_dir)
            status[exp_name] = "completed"
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            status[exp_name] = "failed"
            save_status(status_file, status)
            sys.exit(1)
        save_status(status_file, status)

def generate_slurm_jobs(experiments: Dict[str, Any], folds: List[int], config_path: str, output_dir: Path) -> None:
    """Generate SLURM submission scripts for each experiment/fold combo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for exp_name, exp_config in experiments.items():
        for fold in folds:
            script_content = f"""#!/bin/bash
#SBATCH --job-name={exp_name}_f{fold}
#SBATCH --output={output_dir}/{exp_name}_f{fold}.out
#SBATCH --error={output_dir}/{exp_name}_f{fold}.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate monai_env

python -m src.training.run_experiment run --experiment {exp_name} --folds {fold}
"""
            script_path = output_dir / f"run_{exp_name}_f{fold}.sh"
            with open(script_path, "w") as f:
                f.write(script_content)
    logger.info(f"Generated SLURM scripts in {output_dir}")

def collect_results(experiments: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
    """
    Collect validation results across all completed experiments.
    Parses checkpoint metrics or evaluation summary JSONs from output_dir.
    Does NOT invent or hardcode metrics.
    """
    logger.info(f"Collecting results from {output_dir}...")
    records = []
    
    for exp_name, exp_cfg in experiments.items():
        exp_dir = output_dir / exp_name
        
        # Check for aggregate evaluation metrics JSON
        summary_file = exp_dir / "val_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    metrics = json.load(f)
                records.append({
                    "Model": exp_name,
                    "Framework": exp_cfg.get("framework"),
                    "Mean_Dice": metrics.get("mean_dice", float("nan")),
                    "Liver_Dice": metrics.get("liver_dice", float("nan")),
                    "Right_Kidney_Dice": metrics.get("right_kidney_dice", float("nan")),
                    "Left_Kidney_Dice": metrics.get("left_kidney_dice", float("nan")),
                    "Status": "Evaluated"
                })
                continue
            except Exception as e:
                logger.warning(f"Could not parse {summary_file}: {e}")
                
        # Check fold checkpoints for best metrics
        fold_metrics = []
        for fold in range(5):
            ckpt_path = exp_dir / f"fold_{fold}" / "model_best.pth"
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(str(ckpt_path), map_location="cpu")
                    best_m = ckpt.get("best_metric", float("nan"))
                    fold_metrics.append(best_m)
                except Exception:
                    pass
                    
        if fold_metrics:
            records.append({
                "Model": exp_name,
                "Framework": exp_cfg.get("framework"),
                "Mean_Dice": float(np.mean(fold_metrics)),
                "Liver_Dice": float("nan"),
                "Right_Kidney_Dice": float("nan"),
                "Left_Kidney_Dice": float("nan"),
                "Status": f"Completed ({len(fold_metrics)}/5 folds)"
            })
        else:
            records.append({
                "Model": exp_name,
                "Framework": exp_cfg.get("framework"),
                "Mean_Dice": float("nan"),
                "Liver_Dice": float("nan"),
                "Right_Kidney_Dice": float("nan"),
                "Left_Kidney_Dice": float("nan"),
                "Status": "Pending / Incomplete"
            })
            
    df = pd.DataFrame(records)
    return df

def generate_comparison_table(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison table with Dice per organ per model."""
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison table to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Master experiment orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--experiment", type=str, required=True, choices=list(EXPERIMENTS.keys()))
    run_parser.add_argument("--folds", type=str, required=True, help="Comma-separated folds")
    run_parser.add_argument("--config", type=str, default="configs/base.yaml")
    run_parser.add_argument("--output-dir", type=str, default="results")
    
    run_all_parser = subparsers.add_parser("run-all")
    run_all_parser.add_argument("--config", type=str, default="configs/base.yaml")
    run_all_parser.add_argument("--output-dir", type=str, default="results")
    
    slurm_parser = subparsers.add_parser("generate-slurm")
    slurm_parser.add_argument("--output-dir", type=str, required=True)
    
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == "run":
        folds = [int(f) for f in args.folds.split(",")]
        run_single_experiment(args.experiment, EXPERIMENTS[args.experiment], folds, args.config, Path(args.output_dir))
    elif args.command == "run-all":
        run_all_experiments(EXPERIMENTS, [0, 1, 2, 3, 4], args.config, Path(args.output_dir))
    elif args.command == "generate-slurm":
        generate_slurm_jobs(EXPERIMENTS, [0, 1, 2, 3, 4], "configs/base.yaml", Path(args.output_dir))
    elif args.command == "collect":
        df = collect_results(EXPERIMENTS, Path(args.output).parent)
        generate_comparison_table(df, Path(args.output))

if __name__ == "__main__":
    main()
