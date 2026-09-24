"""
Report generation module.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

def generate_volume_report(
    volumes: Dict[str, float], 
    output_path: Path, 
    format: str = 'txt',
    donor_weight_kg: Optional[float] = None
) -> None:
    """
    Generate a text report summarizing the volumetric findings.
    """
    total_volume = sum(volumes.values())
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        "========================================",
        " VOLUMETRY SEGMENTATION REPORT",
        "========================================",
        f"Date: {date_str}",
        "",
        "Organ Volumes:",
        "----------------------------------------",
    ]
    
    for organ, vol in volumes.items():
        pct = (vol / total_volume * 100) if total_volume > 0 else 0
        report_lines.append(f"{organ.replace('_', ' ').title()}: {vol:.2f} mL ({pct:.1f}%)")
        
    report_lines.append("----------------------------------------")
    
    if donor_weight_kg is not None and "liver_ml" in volumes:
        # standard assumption: 1 mL of liver ~ 1 gram
        gbwr = (volumes["liver_ml"] / (donor_weight_kg * 1000)) * 100
        report_lines.append(f"Estimated GBWR (Graft-to-Recipient Weight Ratio) assuming weight={donor_weight_kg}kg:")
        report_lines.append(f"GBWR: {gbwr:.2f}%")
        report_lines.append("----------------------------------------")
        
    report_lines.extend([
        "",
        "Clinical Context Notes:",
        "- Liver volumes typically range from 1200-1600 mL in healthy adults.",
        "- Kidney volumes typically range from 120-170 mL in healthy adults.",
        "",
        "DISCLAIMER: This is a research tool, NOT a clinical device. ",
        "Volumes should be verified by a medical professional.",
        "========================================",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Report generated at {output_path}")

def generate_evaluation_report(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate an evaluation report comparing models, with optional plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "evaluation_summary.csv"
    metrics_df.to_csv(summary_path, index=False)
    logger.info(f"Saved evaluation metrics to {summary_path}")
    
    # Identify the best model based on a chosen metric, e.g., mean Dice
    if "Dice" in metrics_df.columns and "Model" in metrics_df.columns:
        best_model = metrics_df.groupby("Model")["Dice"].mean().idxmax()
        logger.info(f"Best model based on mean Dice score: {best_model}")
        
    # In a full implementation, you would generate Bland-Altman plots here using matplotlib/seaborn
    # and save them to output_dir
    pass

if __name__ == "__main__":
    pass
