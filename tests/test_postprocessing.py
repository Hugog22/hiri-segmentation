"""
Tests for postprocessing module: connected components, hole filling, small component removal.
"""
import numpy as np
import pytest

from src.inference.postprocessing import (
    compute_component_volumes,
    fill_holes_3d,
    keep_largest_component,
    postprocess_segmentation,
    remove_small_components,
)


class TestConnectedComponents:
    def test_keep_largest_component(self):
        # Create mask with one big blob (1000 voxels) and one small blob (50 voxels)
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[2:12, 2:12, 2:12] = 1   # 10x10x10 = 1000 voxels
        mask[20:25, 20:25, 20:22] = 1 # 5x5x2 = 50 voxels

        result = keep_largest_component(mask, n_keep=1)
        assert np.sum(result) == 1000, f"Expected 1000 voxels, got {np.sum(result)}"
        assert np.all(result[20:25, 20:25, 20:22] == 0), "Small component should be removed"

    def test_remove_small_components(self):
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[2:12, 2:12, 2:12] = 1   # 1000 voxels
        mask[20:22, 20:22, 20:22] = 1 # 8 voxels

        result = remove_small_components(mask, min_voxels=50)
        assert np.sum(result) == 1000
        assert np.all(result[20:22, 20:22, 20:22] == 0)


class TestFillHoles:
    def test_fill_hollow_sphere(self):
        # Create a hollow cube: outer 6x6x6, inner 2x2x2 hollow
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask[4:6, 4:6, 4:6] = 0  # hole of 8 voxels

        filled = fill_holes_3d(mask)
        assert np.sum(filled) == 6 * 6 * 6, "Hole should be completely filled"
        assert np.all(filled[4:6, 4:6, 4:6] == 1)


class TestComputeComponentVolumes:
    def test_volume_accuracy_ml(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1  # 10x10x10 = 1000 voxels
        spacing = (1.0, 1.0, 1.0) # 1 mm^3 per voxel

        # 1000 voxels * 1 mm^3 = 1000 mm^3 = 1.0 mL
        volumes = compute_component_volumes(mask, spacing)
        assert len(volumes) == 1
        assert np.isclose(volumes[0], 1.0, atol=1e-3)


class TestPostprocessSegmentation:
    def test_postprocess_multiclass(self):
        seg = np.zeros((30, 30, 30), dtype=np.uint8)
        # Liver (1) with small satellite
        seg[2:12, 2:12, 2:12] = 1
        seg[15:17, 15:17, 15:17] = 1 # small satellite

        # Right Kidney (2)
        seg[5:10, 20:25, 5:10] = 2

        # Left Kidney (3)
        seg[20:25, 20:25, 20:25] = 3

        config = {
            "inference": {
                "postprocessing": {
                    "min_volume_threshold": 20
                }
            }
        }
        spacing = (1.0, 1.0, 1.0)
        cleaned = postprocess_segmentation(seg, spacing, config)

        # Satellite of liver should be removed
        assert not np.any(cleaned[15:17, 15:17, 15:17] == 1)
        # Major organs preserved
        assert np.any(cleaned == 1)
        assert np.any(cleaned == 2)
        assert np.any(cleaned == 3)
