"""
Ensemble module for combining predictions from multiple models/folds.
Handles softmax probability maps and logits.
"""
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import nibabel as nib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_prob_map(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        return data[data.files[0]]
    elif path.name.endswith(".nii.gz"):
        return nib.load(path).get_fdata()
    else:
        raise ValueError(f"Unsupported file format: {path}")

def save_prediction(data: np.ndarray, path: Path, ref_path: Path) -> None:
    ref_img = nib.load(ref_path)
    img = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(img, path)

def ensemble_predictions_probability(pred_paths: List[Path], output_path: Path, weights: Optional[List[float]] = None) -> None:
    """Compute weighted average of probabilities and take argmax."""
    if weights is None:
        weights = [1.0 / len(pred_paths)] * len(pred_paths)
        
    avg_probs = None
    for path, weight in zip(pred_paths, weights):
        probs = load_prob_map(path)
        if avg_probs is None:
            avg_probs = probs * weight
        else:
            avg_probs += probs * weight
            
    final_pred = np.argmax(avg_probs, axis=0)  # Assuming shape is (C, X, Y, Z)
    save_prediction(final_pred, output_path, pred_paths[0])
    logger.info(f"Saved ensemble prediction to {output_path}")

def ensemble_from_logits(logit_paths: List[Path], output_path: Path, weights: Optional[List[float]] = None) -> None:
    """Compute weighted average of probabilities from logits and take argmax."""
    def softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)
        
    if weights is None:
        weights = [1.0 / len(logit_paths)] * len(logit_paths)
        
    avg_probs = None
    for path, weight in zip(logit_paths, weights):
        logits = load_prob_map(path)
        probs = softmax(logits)
        if avg_probs is None:
            avg_probs = probs * weight
        else:
            avg_probs += probs * weight
            
    final_pred = np.argmax(avg_probs, axis=0)
    save_prediction(final_pred, output_path, logit_paths[0])
    logger.info(f"Saved ensemble from logits to {output_path}")

def ensemble_folds(fold_dirs: List[Path], output_dir: Path, method: str = 'mean') -> None:
    """Combine predictions from multiple folds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Need at least one valid file to start
    if not fold_dirs:
        return
        
    sample_files = list(fold_dirs[0].glob("*.npz")) + list(fold_dirs[0].glob("*.nii.gz"))
    for file_path in sample_files:
        filename = file_path.name
        pred_paths = [d / filename for d in fold_dirs if (d / filename).exists()]
        if len(pred_paths) != len(fold_dirs):
            logger.warning(f"File {filename} missing in some folds. Skipping.")
            continue
        out_path = output_dir / filename.replace(".npz", ".nii.gz")
        ensemble_predictions_probability(pred_paths, out_path)

def ensemble_models(model_dirs: List[Path], output_dir: Path, model_weights: Optional[List[float]] = None) -> None:
    """Combine predictions from different models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_dirs:
        return
        
    sample_files = list(model_dirs[0].glob("*.npz")) + list(model_dirs[0].glob("*.nii.gz"))
    for file_path in sample_files:
        filename = file_path.name
        pred_paths = [d / filename for d in model_dirs if (d / filename).exists()]
        if len(pred_paths) != len(model_dirs):
            logger.warning(f"File {filename} missing in some models. Skipping.")
            continue
        out_path = output_dir / filename.replace(".npz", ".nii.gz")
        ensemble_predictions_probability(pred_paths, out_path, model_weights)

def save_softmax_predictions(model: Any, dataloader: Any, output_dir: Path, device: Any, sliding_window_params: dict) -> None:
    """Run inference and save softmax probability maps."""
    pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble predictions")
    parser.add_argument("--pred-dirs", nargs="+", required=True, help="Directories with predictions")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--method", type=str, default="mean", choices=["mean"])
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for models")
    
    args = parser.parse_args()
    
    pred_dirs = [Path(d) for d in args.pred_dirs]
    out_dir = Path(args.output_dir)
    
    if args.weights:
        ensemble_models(pred_dirs, out_dir, args.weights)
    else:
        ensemble_folds(pred_dirs, out_dir, args.method)

if __name__ == "__main__":
    main()
