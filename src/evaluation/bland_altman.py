"""
Bland-Altman analysis module for volumetry comparisons.
Includes statistical computation and plotting functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    cov = np.cov(y_true, y_pred)[0][1]
    
    numerator = 2 * cov
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    
    return float(numerator / denominator)

def bland_altman_analysis(measured: np.ndarray, reference: np.ndarray) -> dict:
    """Compute statistics for Bland-Altman analysis."""
    measured = np.asarray(measured)
    reference = np.asarray(reference)
    
    diff = measured - reference
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    lower_loa = mean_diff - 1.96 * std_diff
    upper_loa = mean_diff + 1.96 * std_diff
    
    r, p = stats.pearsonr(measured, reference)
    ccc = concordance_correlation_coefficient(reference, measured)
    
    return {
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "lower_loa": float(lower_loa),
        "upper_loa": float(upper_loa),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "ccc": float(ccc)
    }

def plot_bland_altman(measured: np.ndarray, reference: np.ndarray, organ_name: str, output_path: Path, units: str = 'mL') -> None:
    """Create a publication-quality Bland-Altman plot."""
    measured = np.asarray(measured)
    reference = np.asarray(reference)
    
    means = (measured + reference) / 2.0
    diffs = measured - reference
    
    stats_dict = bland_altman_analysis(measured, reference)
    mean_diff = stats_dict["mean_diff"]
    lower_loa = stats_dict["lower_loa"]
    upper_loa = stats_dict["upper_loa"]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.6, edgecolors='k')
    
    # Plot lines
    plt.axhline(mean_diff, color='red', linestyle='--', linewidth=2, label='Mean Bias')
    plt.axhline(lower_loa, color='blue', linestyle=':', linewidth=2, label='-1.96 SD')
    plt.axhline(upper_loa, color='blue', linestyle=':', linewidth=2, label='+1.96 SD')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Annotations
    plt.text(np.max(means)*0.8, mean_diff + (np.max(diffs)*0.02), f'Bias: {mean_diff:.2f}', color='red')
    plt.text(np.max(means)*0.8, upper_loa + (np.max(diffs)*0.02), f'+1.96 SD: {upper_loa:.2f}', color='blue')
    plt.text(np.max(means)*0.8, lower_loa + (np.max(diffs)*0.02), f'-1.96 SD: {lower_loa:.2f}', color='blue')
    
    plt.title(f'Bland-Altman Plot: {organ_name}')
    plt.xlabel(f'Mean Volume ({units})')
    plt.ylabel(f'Volume Difference ({units})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_correlation(measured: np.ndarray, reference: np.ndarray, organ_name: str, output_path: Path) -> None:
    """Create a scatter plot with regression and identity lines."""
    measured = np.asarray(measured)
    reference = np.asarray(reference)
    
    stats_dict = bland_altman_analysis(measured, reference)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reference, measured, alpha=0.6, edgecolors='k')
    
    # Identity line
    min_val = min(np.min(reference), np.min(measured))
    max_val = max(np.max(reference), np.max(measured))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Identity Line')
    
    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(reference, measured)
    reg_x = np.array([min_val, max_val])
    reg_y = slope * reg_x + intercept
    plt.plot(reg_x, reg_y, 'r-', linewidth=2, label=f'Regression (R²={r_value**2:.3f})')
    
    plt.title(f'Volume Correlation: {organ_name}\nCCC = {stats_dict["ccc"]:.3f}')
    plt.xlabel('Reference Volume (mL)')
    plt.ylabel('Measured Volume (mL)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
