"""
Pytest configuration and shared fixtures for the HI&RI project tests.
"""
import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
import json

def make_synthetic_image(shape: tuple[int, int, int], spacing: tuple[float, float, float], affine: np.ndarray, hu_range: tuple[float, float]) -> nib.Nifti1Image:
    """Create a synthetic NIfTI image with random values in hu_range."""
    data = np.random.uniform(hu_range[0], hu_range[1], size=shape).astype(np.float32)
    return nib.Nifti1Image(data, affine)

def make_synthetic_mask(shape: tuple[int, int, int], spacing: tuple[float, float, float], affine: np.ndarray, label_positions: dict) -> nib.Nifti1Image:
    """Create a synthetic NIfTI mask with spheres at given label positions."""
    data = np.zeros(shape, dtype=np.uint8)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    for label_id, (cx, cy, cz, r) in label_positions.items():
        mask = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2
        data[mask] = label_id
    return nib.Nifti1Image(data, affine)

def save_nifti(image: nib.Nifti1Image, path: Path) -> None:
    """Save NIfTI image to path."""
    nib.save(image, str(path))

@pytest.fixture
def tmp_nifti_image(tmp_path: Path) -> Path:
    """Provides a path to a temporary NIfTI image."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 2.0)
    affine = np.diag([-1.0, -1.0, 2.0, 1.0])
    image = make_synthetic_image(shape, spacing, affine, (-1000, 1000))
    path = tmp_path / "image.nii.gz"
    save_nifti(image, path)
    return path

@pytest.fixture
def tmp_nifti_mask(tmp_path: Path) -> Path:
    """Provides a path to a temporary NIfTI mask."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 2.0)
    affine = np.diag([-1.0, -1.0, 2.0, 1.0])
    label_positions = {
        1: (32, 32, 32, 10),
        2: (20, 40, 20, 5),
        3: (44, 40, 20, 5)
    }
    mask = make_synthetic_mask(shape, spacing, affine, label_positions)
    path = tmp_path / "mask.nii.gz"
    save_nifti(mask, path)
    return path

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Creates a mock dataset directory."""
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    return data_dir
