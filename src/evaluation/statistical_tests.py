"""
Statistical analysis module for comparing segmentation models.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Compute bootstrap confidence intervals for the mean."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]  # Remove NaNs
    
    if len(values) == 0:
        return np.nan, np.nan, np.nan
        
    rng = np.random.default_rng(seed)
    bootstrap_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means[i] = np.mean(sample)
        
    mean_val = np.mean(values)
    alpha = (1.0 - ci) / 2.0
    lower_ci = float(np.percentile(bootstrap_means, alpha * 100))
    upper_ci = float(np.percentile(bootstrap_means, (1.0 - alpha) * 100))
    
    return float(mean_val), lower_ci, upper_ci

def paired_wilcoxon_test(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    """Perform paired Wilcoxon signed-rank test."""
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    
    # Need to handle NaNs
    mask = ~np.isnan(values_a) & ~np.isnan(values_b)
    v_a = values_a[mask]
    v_b = values_b[mask]
    
    if len(v_a) < 2:
        return np.nan, np.nan
        
    diff = v_a - v_b
    if np.all(diff == 0):
        return 0.0, 1.0
        
    stat, p_value = stats.wilcoxon(v_a, v_b)
    return float(stat), float(p_value)

def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Apply Bonferroni correction for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return []
    
    corrected = []
    for p in p_values:
        if np.isnan(p):
            corrected.append((np.nan, False))
        else:
            p_adj = min(1.0, p * m)
            corrected.append((float(p_adj), p_adj < alpha))
    return corrected

def compare_models(results_a: pd.DataFrame, results_b: pd.DataFrame, metric: str = 'dice', organs: list[str] | None = None) -> pd.DataFrame:
    """Compare two models across organs with Wilcoxon tests and Bootstrap CIs."""
    if organs is None:
        organs = list(set(results_a['organ']).intersection(results_b['organ']))
        
    comparison = []
    p_values = []
    
    for organ in organs:
        df_a = results_a[results_a['organ'] == organ].sort_values('case_id')
        df_b = results_b[results_b['organ'] == organ].sort_values('case_id')
        
        merged = pd.merge(df_a, df_b, on='case_id', suffixes=('_a', '_b'))
        if merged.empty:
            continue
            
        vals_a = merged[f'{metric}_a'].values
        vals_b = merged[f'{metric}_b'].values
        
        mean_a, low_a, up_a = bootstrap_ci(vals_a)
        mean_b, low_b, up_b = bootstrap_ci(vals_b)
        
        stat, p_val = paired_wilcoxon_test(vals_a, vals_b)
        p_values.append(p_val)
        
        comparison.append({
            'organ': organ,
            'metric': metric,
            'mean_A': mean_a,
            'ci_lower_A': low_a,
            'ci_upper_A': up_a,
            'mean_B': mean_b,
            'ci_lower_B': low_b,
            'ci_upper_B': up_b,
            'mean_diff': mean_b - mean_a,
            'wilcoxon_stat': stat,
            'raw_p_value': p_val
        })
        
    # Apply Bonferroni
    corrected = bonferroni_correction(p_values)
    for i, c in enumerate(corrected):
        comparison[i]['bonferroni_p'] = c[0]
        comparison[i]['significant'] = c[1]
        
    return pd.DataFrame(comparison)
