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
    """Set nnUNet_raw, nnUNet_preprocessed, nnUNet_results environment variables if not already set."""
    if "nnUNet_raw" not in os.environ:
        os.environ["nnUNet_raw"] = str(base_dir / "nnUNet_raw")
    if "nnUNet_preprocessed" not in os.environ:
        os.environ["nnUNet_preprocessed"] = str(base_dir / "nnUNet_preprocessed")
    if "nnUNet_results" not in os.environ:
        os.environ["nnUNet_results"] = str(base_dir / "nnUNet_results")
    for var in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
    logger.info(f"nnU-Net environment configured: raw={os.environ['nnUNet_raw']}")

def _run_cmd(cmd: List[str]) -> None:
    """Run a shell command with logging."""
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        sys.exit(e.returncode)

def run_planning(dataset_id: int, planner: str = "nnUNetPlannerResEncL", verify_integrity: bool = False) -> None:
    """Run nnUNetv2_plan_and_preprocess with automatic fallback to standard planner."""
    # Ensure NibabelIO is configured to avoid SimpleITK orthonormal cosine warnings
    import json
    raw_dir = Path(os.environ.get("nnUNet_raw", Path.cwd() / "nnUNet_raw"))
    ds_json_path = raw_dir / f"Dataset{dataset_id:03d}_HIRI" / "dataset.json"
    if ds_json_path.exists():
        try:
            with open(ds_json_path, "r") as f:
                d_meta = json.load(f)
            if d_meta.get("overwrite_image_reader_writer") != "NibabelIO":
                d_meta["overwrite_image_reader_writer"] = "NibabelIO"
                with open(ds_json_path, "w") as f:
                    json.dump(d_meta, f, indent=4)
                logger.info("Configured dataset.json with overwrite_image_reader_writer: NibabelIO")
        except Exception as e:
            logger.warning(f"Could not update dataset.json reader: {e}")

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-pl", planner,
    ]
    if verify_integrity:
        cmd.append("--verify_dataset_integrity")

    try:
        _run_cmd(cmd)
    except SystemExit:
        if planner != "nnUNetPlanner":
            logger.warning(f"Planner '{planner}' failed or unavailable. Falling back to default 'nnUNetPlanner'...")
            fallback_cmd = [
                "nnUNetv2_plan_and_preprocess",
                "-d", str(dataset_id),
                "-pl", "nnUNetPlanner",
            ]
            if verify_integrity:
                fallback_cmd.append("--verify_dataset_integrity")
            _run_cmd(fallback_cmd)
        else:
            raise

def resolve_plans_name(dataset_id: int, planner_or_plans: str) -> str:
    """Resolves planner class name to the actual plans identifier in nnUNet_preprocessed."""
    mapping = {
        "nnUNetPlannerResEncL": "nnUNetResEncUNetLPlans",
        "nnUNetPlannerResEncXL": "nnUNetResEncUNetXLPlans",
        "nnUNetPlanner": "nnUNetPlans",
    }
    candidate = mapping.get(planner_or_plans, planner_or_plans)
    prep_dir = Path(os.environ.get("nnUNet_preprocessed", Path.cwd() / "nnUNet_preprocessed")) / f"Dataset{dataset_id:03d}_HIRI"
    if (prep_dir / f"{candidate}.json").exists():
        return candidate
    found_plans = list(prep_dir.glob("*Plans.json"))
    if found_plans:
        return found_plans[0].stem
    return candidate


def run_training(dataset_id: int, config: str, fold: int, trainer: str = "nnUNetTrainer", 
                 planner: str = "nnUNetPlannerResEncL", resume: bool = False) -> None:
    """Run nnUNetv2_train."""
    plans_name = resolve_plans_name(dataset_id, planner)
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        config,
        str(fold),
        "-tr", trainer,
        "-p", plans_name
    ]
    if resume:
        cmd.append("--c")
    _run_cmd(cmd)

def run_prediction(dataset_id: int, config: str, fold: Union[int, str], input_dir: Path, 
                   output_dir: Path, planner: str = "nnUNetPlannerResEncL") -> None:
    """Run nnUNetv2_predict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_name = resolve_plans_name(dataset_id, planner)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-f", str(fold),
        "-p", plans_name
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
    plan_parser.add_argument("--verify", action="store_true", help="Run verify_dataset_integrity")
    
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
        run_planning(args.dataset_id, args.planner, args.verify)
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
