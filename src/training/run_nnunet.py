"""
Wrapper script for nnU-Net v2 training and prediction.
Integrates nnU-Net into the HI&RI project workflow.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_nnunet_env(base_dir: Path) -> None:
    """Set nnUNet_raw, nnUNet_preprocessed, nnUNet_results environment variables."""
    nnunet_dir = base_dir / "data" / "nnunet"
    os.environ["nnUNet_raw"] = str(nnunet_dir / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(nnunet_dir / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_dir / "nnUNet_results")
    logger.info("Set nnU-Net environment variables.")

def _run_cmd(cmd: List[str]) -> None:
    """Run a shell command with logging."""
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        sys.exit(e.returncode)

def run_planning(dataset_id: int, planner: str = "nnUNetPlannerResEncL") -> None:
    """Run nnUNetv2_plan_and_preprocess."""
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-pl", planner,
        "--verify_dataset_integrity"
    ]
    _run_cmd(cmd)

def run_training(dataset_id: int, config: str, fold: int, trainer: str = "nnUNetTrainer", 
                 planner: str = "nnUNetPlannerResEncL", resume: bool = False) -> None:
    """Run nnUNetv2_train."""
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        config,
        str(fold),
        "-tr", trainer,
        "-p", planner
    ]
    if resume:
        cmd.append("--c")
    _run_cmd(cmd)

def run_prediction(dataset_id: int, config: str, fold: Union[int, str], input_dir: Path, 
                   output_dir: Path, planner: str = "nnUNetPlannerResEncL") -> None:
    """Run nnUNetv2_predict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-f", str(fold),
        "-p", planner
    ]
    _run_cmd(cmd)

def run_ensemble_predictions(dataset_id: int, configs: List[str], folds: List[int], 
                             input_dir: Path, output_dir: Path, planner: str) -> None:
    """Run nnUNetv2_ensemble."""
    logger.info("Ensemble not directly supported by this wrapper yet; use ensemble.py.")
    pass

def find_best_configuration(dataset_id: int) -> None:
    """Run nnUNetv2_find_best_configuration."""
    cmd = [
        "nnUNetv2_find_best_configuration",
        "-d", str(dataset_id),
        "-c", "3d_fullres", "3d_cascade_fullres"
    ]
    _run_cmd(cmd)

def main() -> None:
    parser = argparse.ArgumentParser(description="nnU-Net wrapper for HI&RI")
    parser.add_argument("--base-dir", type=str, default=".", help="Base project directory")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--dataset-id", type=int, required=True)
    plan_parser.add_argument("--planner", type=str, default="nnUNetPlannerResEncL")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-id", type=int, required=True)
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--fold", type=int, required=True)
    train_parser.add_argument("--planner", type=str, default="nnUNetPlannerResEncL")
    train_parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    train_parser.add_argument("--resume", action="store_true")
    
    all_folds_parser = subparsers.add_parser("all-folds")
    all_folds_parser.add_argument("--dataset-id", type=int, required=True)
    all_folds_parser.add_argument("--config", type=str, required=True)
    all_folds_parser.add_argument("--planner", type=str, default="nnUNetPlannerResEncL")
    all_folds_parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--dataset-id", type=int, required=True)
    predict_parser.add_argument("--config", type=str, required=True)
    predict_parser.add_argument("--fold", type=str, required=True)
    predict_parser.add_argument("--input-dir", type=str, required=True)
    predict_parser.add_argument("--output-dir", type=str, required=True)
    predict_parser.add_argument("--planner", type=str, default="nnUNetPlannerResEncL")
    
    best_parser = subparsers.add_parser("find-best")
    best_parser.add_argument("--dataset-id", type=int, required=True)
    
    args = parser.parse_args()
    setup_nnunet_env(Path(args.base_dir).resolve())
    
    if args.command == "plan":
        run_planning(args.dataset_id, args.planner)
    elif args.command == "train":
        run_training(args.dataset_id, args.config, args.fold, args.trainer, args.planner, args.resume)
    elif args.command == "all-folds":
        for fold in range(5):
            run_training(args.dataset_id, args.config, fold, args.trainer, args.planner)
    elif args.command == "predict":
        run_prediction(args.dataset_id, args.config, args.fold, Path(args.input_dir), Path(args.output_dir), args.planner)
    elif args.command == "find-best":
        find_best_configuration(args.dataset_id)

if __name__ == "__main__":
    main()
