"""
Tests for volume calculation.
"""
import pytest
import numpy as np

def calculate_volumes(mask: np.ndarray, spacing: tuple[float, float, float], label_map: dict) -> dict:
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    volumes = {}
    for name, label in label_map.items():
        if label == 0:
            continue
        count = np.sum(mask == label)
        volumes[name] = (count * voxel_vol) / 1000.0 # to mL
    return volumes

def test_known_volume():
    R = 10
    shape = (30, 30, 30)
    spacing = (1.0, 1.0, 1.0)
    
    mask = np.zeros(shape)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cx, cy, cz = 15, 15, 15
    mask_sphere = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= R**2
    mask[mask_sphere] = 1
    
    vols = calculate_volumes(mask, spacing, {'liver': 1})
    computed = vols['liver']
    expected = (4/3 * np.pi * R**3) / 1000.0
    
    assert abs(computed - expected) / expected < 0.05

def test_volume_with_different_spacings():
    mask = np.zeros((10,10,10))
    mask[0:5] = 1
    
    vols1 = calculate_volumes(mask, (1.0, 1.0, 1.0), {'liver': 1})
    vols2 = calculate_volumes(mask, (2.0, 2.0, 2.0), {'liver': 1})
    
    assert vols2['liver'] == 8 * vols1['liver']

def test_volume_uses_original_spacing():
    mask = np.zeros((10,10,10))
    mask[0:5] = 1
    orig_spacing = (1.5, 1.5, 1.5)
    vols = calculate_volumes(mask, orig_spacing, {'liver': 1})
    expected = (np.sum(mask == 1) * 1.5**3) / 1000.0
    assert vols['liver'] == expected

def test_volume_units_ml():
    mask = np.zeros((2,2,2))
    mask[:] = 1
    vols = calculate_volumes(mask, (10.0, 10.0, 10.0), {'liver': 1})
    assert vols['liver'] == 8.0

def test_empty_label_volume_zero():
    mask = np.zeros((10,10,10))
    vols = calculate_volumes(mask, (1.0, 1.0, 1.0), {'liver': 1})
    assert vols['liver'] == 0.0

def test_volume_error_computation():
    pred_vol = 100.0
    gt_vol = 80.0
    abs_err = abs(pred_vol - gt_vol)
    rel_err = abs_err / gt_vol
    assert abs_err == 20.0
    assert rel_err == 0.25

def test_all_volumes_returns_all_organs():
    mask = np.zeros((5,5,5))
    mask[0,0,0] = 1
    labels = {'liver': 1, 'right_kidney': 2, 'left_kidney': 3}
    vols = calculate_volumes(mask, (1.0, 1.0, 1.0), labels)
    assert set(vols.keys()) == {'liver', 'right_kidney', 'left_kidney'}
