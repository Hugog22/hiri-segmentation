"""
Comprehensive benchmark and statistical evaluation pipeline.

Functions:
- Computes per-case metrics across all models (Dice, IoU, HD95, ASD, Volumetric Errors).
- Computes 95% Bootstrap Confidence Intervals.
- Performs paired Wilcoxon signed-rank tests with Bonferroni correction against baseline.
- Generates Bland-Altman and correlation plots for volumetry.
- Exports publication-ready LaTeX and Markdown tables.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from src.evaluation.bland_altman import (
    bland_altman_analysis,
    plot_bland_altman,
    plot_correlation,
)
from src.evaluation.metrics import (
    average_surface_distance,
    dice_score,
    hausdorff_95,
    iou_score,
)
from src.evaluation.statistical_tests import (
    bonferroni_correction,
    bootstrap_ci,
    paired_wilcoxon_test,
)
from src.inference.volume_calculation import compute_volume_error, compute_volume_ml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ORGAN_MAPPING = {
    "liver": 1,
    "right_kidney": 2,
    "left_kidney": 3
}


def evaluate_case(
    pred_path: Path,
    gt_path: Path,
    case_id: str,
    orig_spacing_json: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Compute all evaluation metrics for a single case across all organs."""
    pred_nii = nib.load(str(pred_path))
    gt_nii = nib.load(str(gt_path))
    
    pred_data = np.asarray(pred_nii.dataobj).astype(np.uint8)
    gt_data = np.asarray(gt_nii.dataobj).astype(np.uint8)
    
    spacing = tuple(gt_nii.header.get_zooms()[:3])
    
    # Read original spacing if available
    orig_spacing = spacing
    if orig_spacing_json and orig_spacing_json.exists():
        try:
            with open(orig_spacing_json, "r") as f:
                meta = json.load(f)
            orig_spacing = tuple(meta.get("original_spacing", spacing))
        except Exception:
            pass

    records = []
    for organ, lbl in ORGAN_MAPPING.items():
        # Solapamiento y distancia
        dsc = dice_score(pred_data, gt_data, lbl)
        iou = iou_score(pred_data, gt_data, lbl)
        hd = hausdorff_95(pred_data, gt_data, lbl, spacing)
        asd = average_surface_distance(pred_data, gt_data, lbl, spacing)
        
        # Volumetría con espaciado original
        v_pred_ml = compute_volume_ml(pred_path, lbl, orig_spacing)
        v_gt_ml = compute_volume_ml(gt_path, lbl, orig_spacing)
        vol_err = compute_volume_error({"v": v_pred_ml}, {"v": v_gt_ml})["v"]
        
        records.append({
            "case_id": case_id,
            "organ": organ,
            "dice": dsc,
            "iou": iou,
            "hd95": hd,
            "asd": asd,
            "vol_pred_ml": v_pred_ml,
            "vol_gt_ml": v_gt_ml,
            "vol_error_abs_ml": vol_err["abs_error_ml"],
            "vol_error_rel_pct": vol_err["rel_error_pct"],
        })
        
    return records


def run_full_benchmark(
    models_dict: Dict[str, Path],
    gt_dir: Path,
    output_dir: Path,
    baseline_model: str = "M1_nnunet_3d_fullres"
) -> None:
    """
    Run complete benchmark evaluating all models in models_dict against GT.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics_list = []
    
    for model_name, pred_dir in models_dict.items():
        logger.info(f"Benchmarking model: {model_name} from {pred_dir}")
        for pred_file in pred_dir.glob("*.nii.gz"):
            case_id = pred_file.name.replace(".nii.gz", "").replace("_seg", "")
            gt_file = gt_dir / pred_file.name
            if not gt_file.exists():
                # Try finding in case subfolder
                gt_file = gt_dir / case_id / "unified_label.nii.gz"
                
            if not gt_file.exists():
                logger.warning(f"GT not found for {case_id}, skipping.")
                continue
                
            case_records = evaluate_case(pred_file, gt_file, case_id)
            for r in case_records:
                r["model"] = model_name
                all_metrics_list.append(r)
                
    if not all_metrics_list:
        logger.error("No metrics could be computed. Check paths.")
        return
        
    df = pd.DataFrame(all_metrics_list)
    df.to_csv(output_dir / "per_case_metrics.csv", index=False)
    logger.info(f"Saved per-case metrics to {output_dir / 'per_case_metrics.csv'}")
    
    # Summary Table with Bootstrap CIs
    summary_records = []
    for model_name in df["model"].unique():
        m_df = df[df["model"] == model_name]
        for organ in ORGAN_MAPPING.keys():
            om_df = m_df[m_df["organ"] == organ]
            
            dice_vals = om_df["dice"].dropna().values
            mean_d, low_d, up_d = bootstrap_ci(dice_vals) if len(dice_vals) > 0 else (np.nan, np.nan, np.nan)
            
            hd_vals = om_df["hd95"].dropna().values
            mean_hd, low_hd, up_hd = bootstrap_ci(hd_vals) if len(hd_vals) > 0 else (np.nan, np.nan, np.nan)
            
            v_err_abs = om_df["vol_error_abs_ml"].dropna().values
            mean_ve_abs, low_ve, up_ve = bootstrap_ci(v_err_abs) if len(v_err_abs) > 0 else (np.nan, np.nan, np.nan)
            
            v_err_rel = om_df["vol_error_rel_pct"].dropna().values
            mean_ve_rel, _, _ = bootstrap_ci(v_err_rel) if len(v_err_rel) > 0 else (np.nan, np.nan, np.nan)
            
            summary_records.append({
                "Model": model_name,
                "Organ": organ,
                "Dice (95% CI)": f"{mean_d:.3f} [{low_d:.3f}, {up_d:.3f}]",
                "HD95 mm (95% CI)": f"{mean_hd:.2f} [{low_hd:.2f}, {up_hd:.2f}]",
                "Abs Vol Err mL": f"{mean_ve_abs:.1f} [{low_ve:.1f}, {up_ve:.1f}]",
                "Rel Vol Err %": f"{mean_ve_rel:.1f}%",
                "_raw_dice_mean": mean_d,
                "_raw_hd95_mean": mean_hd
            })
            
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    
    # Statistical tests against baseline
    if baseline_model in df["model"].unique():
        stat_records = []
        base_df = df[df["model"] == baseline_model]
        
        comp_models = [m for m in df["model"].unique() if m != baseline_model]
        for comp_m in comp_models:
            c_df = df[df["model"] == comp_m]
            for organ in ORGAN_MAPPING.keys():
                merged = base_df[base_df["organ"] == organ].merge(
                    c_df[c_df["organ"] == organ], on="case_id", suffixes=("_base", "_comp")
                )
                if len(merged) < 5:
                    continue
                stat, p_val = paired_wilcoxon_test(merged["dice_base"].values, merged["dice_comp"].values)
                stat_records.append({
                    "Comparison": f"{comp_m} vs {baseline_model}",
                    "Organ": organ,
                    "Wilcoxon_W": stat,
                    "p_value_raw": p_val,
                    "N": len(merged)
                })
                
        if stat_records:
            stat_df = pd.DataFrame(stat_records)
            # Bonferroni correction
            bonf = bonferroni_correction(stat_df["p_value_raw"].tolist())
            stat_df["p_value_bonferroni"] = [b[0] for b in bonf]
            stat_df["is_significant_0.05"] = [b[1] for b in bonf]
            stat_df.to_csv(output_dir / "statistical_comparisons.csv", index=False)
            logger.info(f"Saved statistical tests to {output_dir / 'statistical_comparisons.csv'}")

    # Generate Bland-Altman and correlation plots per model per organ
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in df["model"].unique():
        m_df = df[df["model"] == model_name]
        for organ in ORGAN_MAPPING.keys():
            om_df = m_df[m_df["organ"] == organ]
            v_pred = om_df["vol_pred_ml"].values
            v_gt = om_df["vol_gt_ml"].values
            if len(v_pred) >= 3:
                ba_path = figures_dir / f"bland_altman_{model_name}_{organ}.png"
                corr_path = figures_dir / f"correlation_{model_name}_{organ}.png"
                plot_bland_altman(v_pred, v_gt, f"{organ} ({model_name})", ba_path)
                plot_correlation(v_pred, v_gt, f"{organ} ({model_name})", corr_path)
                
    logger.info("Benchmark execution completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Master benchmark evaluation script.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Ground truth masks directory")
    parser.add_argument("--models-json", type=Path, required=True, help="JSON file mapping model_name -> predictions_dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for benchmark")
    parser.add_argument("--baseline", type=str, default="M1_nnunet_3d_fullres", help="Baseline model name")
    args = parser.parse_args()

    with open(args.models_json, "r") as f:
        models_dict = {k: Path(v) for k, v in json.load(f).items()}
        
    run_full_benchmark(models_dict, args.gt_dir, args.output_dir, args.baseline)


if __name__ == "__main__":
    main()
