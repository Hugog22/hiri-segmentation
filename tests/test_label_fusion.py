"""
Tests for label map unification logic.
"""
import pytest
import numpy as np

def convert_labels(mask: np.ndarray, dataset: str, merge_tumors: bool = True) -> np.ndarray:
    out = np.zeros_like(mask)
    if dataset == 'totalsegmentator':
        out[mask == 5] = 1
        out[mask == 2] = 2
        out[mask == 3] = 3
    elif dataset == 'lits':
        out[mask == 1] = 1
        if merge_tumors:
            out[mask == 2] = 1
        else:
            out[mask == 2] = 4
    elif dataset in ['amos', 'btcv']:
        out[mask == 6] = 1
        out[mask == 2] = 2
        out[mask == 3] = 3
    return out

def test_totalseg_conversion():
    mask = np.array([0, 2, 3, 5, 7, 8])
    out = convert_labels(mask, 'totalsegmentator')
    assert set(np.unique(out)) == {0, 1, 2, 3}

def test_lits_conversion_merge_tumors():
    mask = np.array([0, 1, 2])
    out = convert_labels(mask, 'lits', merge_tumors=True)
    assert set(np.unique(out)) == {0, 1}

def test_lits_conversion_keep_tumors():
    mask = np.array([0, 1, 2])
    out = convert_labels(mask, 'lits', merge_tumors=False)
    assert set(np.unique(out)) == {0, 1, 4}

def test_amos_conversion():
    mask = np.array([0, 2, 3, 6, 7, 8])
    out = convert_labels(mask, 'amos')
    assert set(np.unique(out)) == {0, 1, 2, 3}

def test_btcv_conversion():
    mask = np.array([0, 2, 3, 6, 7, 8])
    out = convert_labels(mask, 'btcv')
    assert set(np.unique(out)) == {0, 1, 2, 3}

def test_output_contains_only_valid_labels():
    mask = np.random.randint(0, 100, size=(10,10,10))
    out = convert_labels(mask, 'totalsegmentator')
    assert set(np.unique(out)).issubset({0, 1, 2, 3})

def test_voxel_counts_preserved():
    mask = np.zeros((10,10,10))
    mask[0:2] = 5
    out = convert_labels(mask, 'totalsegmentator')
    assert np.sum(mask == 5) == np.sum(out == 1)

def test_spatial_consistency():
    mask = np.zeros((10,10,10))
    mask[5,5,5] = 5
    out = convert_labels(mask, 'totalsegmentator')
    assert out[5,5,5] == 1
