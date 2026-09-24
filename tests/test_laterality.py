"""
Tests for resolve_kits_laterality module.

Validates kidney laterality resolution from KiTS23 (which labels both
kidneys with value=1). Uses synthetic NIfTI data — no real files needed.
"""
import numpy as np
import pytest

from src.preprocessing.resolve_kits_laterality import resolve_laterality


def _make_affine(spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Create a simple RAS-aligned affine with given voxel spacing."""
    aff = np.eye(4)
    aff[0, 0] = spacing[0]
    aff[1, 1] = spacing[1]
    aff[2, 2] = spacing[2]
    return aff


def _sphere(shape: tuple, center: tuple, radius: float) -> np.ndarray:
    """Create a binary sphere mask."""
    coords = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist_sq = sum((c - ctr) ** 2 for c, ctr in zip(coords, center))
    return (dist_sq <= radius ** 2).astype(np.uint8)


class TestTwoKidneysCorrectAssignment:
    """Standard case: two clearly separated kidneys."""

    def test_right_lower_x_left_higher_x(self):
        """In RAS: lower X = patient's right (label 2), higher X = left (label 3)."""
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        # Right kidney: centered at X=10 (lower X → right in RAS)
        mask[5:15, 25:35, 25:35] = 1
        # Left kidney: centered at X=50 (higher X → left in RAS)
        mask[45:55, 25:35, 25:35] = 1

        affine = _make_affine()
        resolved, report = resolve_laterality(mask, affine)

        assert np.all(resolved[5:15, 25:35, 25:35] == 2), "Lower X should be right kidney (2)"
        assert np.all(resolved[45:55, 25:35, 25:35] == 3), "Higher X should be left kidney (3)"
        assert report['confidence'] == 'high'
        assert report['num_components_major'] == 2

    def test_with_anisotropic_spacing(self):
        """Verify assignment works with non-identity affine."""
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        mask[5:15, 25:35, 25:35] = 1   # right
        mask[45:55, 25:35, 25:35] = 1  # left

        affine = _make_affine(spacing=(0.7, 0.7, 2.5))
        resolved, report = resolve_laterality(mask, affine)

        assert np.all(resolved[5:15, 25:35, 25:35] == 2)
        assert np.all(resolved[45:55, 25:35, 25:35] == 3)


class TestSingleKidney:
    """Single kidney component — should be flagged for review."""

    def test_single_kidney_flagged(self):
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        mask[5:15, 25:35, 25:35] = 1  # Only one kidney

        resolved, report = resolve_laterality(mask, _make_affine())

        assert report['confidence'] == 'low'
        # Should be assigned as either 2 or 3 based on position
        kidney_labels = np.unique(resolved[resolved > 0])
        assert len(kidney_labels) >= 1
        assert 2 in kidney_labels or 3 in kidney_labels

    def test_single_right_kidney(self):
        """Single kidney at negative X (right)."""
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        mask[5:15, 25:35, 25:35] = 1

        # Affine that places voxel X=10 at world X=-20 (right side)
        affine = _make_affine()
        affine[0, 3] = -32  # origin shift so center of volume is at world X=0

        resolved, report = resolve_laterality(mask, affine)
        assert report['confidence'] == 'low'


class TestMultipleFragments:
    """More than 2 components — should merge fragments into nearest kidney."""

    def test_three_components_two_major(self):
        mask = np.zeros((80, 64, 64), dtype=np.int16)
        # Large right kidney
        mask[5:20, 25:35, 25:35] = 1
        # Small fragment near right kidney
        mask[22:24, 28:31, 28:31] = 1
        # Large left kidney
        mask[55:70, 25:35, 25:35] = 1

        resolved, report = resolve_laterality(mask, _make_affine())

        # Both large components should be assigned
        assert np.any(resolved == 2), "Should have right kidney"
        assert np.any(resolved == 3), "Should have left kidney"
        # Small fragment should be merged into nearest (right) kidney
        assert np.all(resolved[22:24, 28:31, 28:31] == 2), \
            "Fragment near right kidney should be merged into right"

    def test_many_small_fragments(self):
        """All fragments below threshold but two dominant ones."""
        mask = np.zeros((80, 64, 64), dtype=np.int16)
        # Two main kidneys (above default MIN=500)
        mask[5:20, 20:35, 20:35] = 1   # ~15*15*15 = 3375 voxels
        mask[55:70, 20:35, 20:35] = 1  # ~15*15*15 = 3375 voxels
        # Tiny fragments
        mask[30:32, 30:32, 30:32] = 1  # 8 voxels
        mask[40:42, 30:32, 30:32] = 1  # 8 voxels

        resolved, report = resolve_laterality(mask, _make_affine())

        assert np.any(resolved == 2)
        assert np.any(resolved == 3)
        # No unresolved kidney label (1) should remain
        assert not np.any(resolved == 1), "No unresolved kidney voxels should remain"


class TestNoKidney:
    """No kidney voxels at all."""

    def test_empty_mask(self):
        mask = np.zeros((32, 32, 32), dtype=np.int16)
        resolved, report = resolve_laterality(mask, _make_affine())
        assert np.array_equal(resolved, np.zeros_like(mask, dtype=np.uint8))
        assert report['confidence'] == 'error'

    def test_mask_with_only_tumors(self):
        """Mask has tumor labels but no kidney label."""
        mask = np.zeros((32, 32, 32), dtype=np.int16)
        mask[5:10, 5:10, 5:10] = 2  # tumor
        mask[20:25, 20:25, 20:25] = 3  # cyst

        resolved, report = resolve_laterality(mask, _make_affine())
        # Tumors and cysts should be preserved with unified labels
        assert np.any(resolved == 5)  # unified tumor
        assert np.any(resolved == 6)  # unified cyst
        assert report['confidence'] == 'error'


class TestHorseshoeKidney:
    """Single connected component spanning both sides (horseshoe kidney)."""

    def test_horseshoe_flagged(self):
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        # Kidney spanning full X range — looks like horseshoe
        mask[5:60, 25:35, 25:35] = 1  # spans 55 voxels in X

        resolved, report = resolve_laterality(mask, _make_affine())

        # Should be flagged as low confidence (single component)
        assert report['confidence'] == 'low'
        # Should still assign some label
        assert np.any((resolved == 2) | (resolved == 3))


class TestWorldCoordinates:
    """Verify centroid calculation uses affine (world coords)."""

    def test_flipped_affine(self):
        """With a left-handed affine (negative X scaling), the assignment
        should still be correct because we use world coordinates."""
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        mask[5:15, 25:35, 25:35] = 1   # voxel X=10
        mask[45:55, 25:35, 25:35] = 1  # voxel X=50

        # Negative X scaling: flips the X axis
        # Voxel X=10 → world X=-10 (right), Voxel X=50 → world X=-50 (more right)
        affine = np.diag([-1.0, 1.0, 1.0, 1.0])

        resolved, report = resolve_laterality(mask, affine)

        # With negative X scaling:
        # voxel X=10 → world X=-10
        # voxel X=50 → world X=-50 (more negative = more right)
        # So the blob at voxel X=45:55 should be RIGHT kidney
        assert np.all(resolved[45:55, 25:35, 25:35] == 2), \
            "With flipped affine, higher voxel X maps to lower world X (right)"
        assert np.all(resolved[5:15, 25:35, 25:35] == 3), \
            "With flipped affine, lower voxel X maps to higher world X (left)"

    def test_offset_origin(self):
        """Affine with non-zero origin offset."""
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        mask[5:15, 25:35, 25:35] = 1
        mask[45:55, 25:35, 25:35] = 1

        affine = _make_affine()
        affine[0, 3] = -100  # shift origin
        affine[1, 3] = -200
        affine[2, 3] = 50

        resolved, report = resolve_laterality(mask, affine)

        # Assignment should be the same regardless of origin offset
        # (only relative X matters for laterality)
        assert np.all(resolved[5:15, 25:35, 25:35] == 2)
        assert np.all(resolved[45:55, 25:35, 25:35] == 3)


class TestPreservesOtherLabels:
    """Tumor and cyst labels should be preserved (remapped to unified)."""

    def test_tumor_preserved(self):
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        # Two kidneys
        mask[5:15, 25:35, 25:35] = 1
        mask[45:55, 25:35, 25:35] = 1
        # Tumor inside right kidney
        mask[8:12, 28:32, 28:32] = 2
        # Cyst near left kidney
        mask[48:52, 28:32, 28:32] = 3

        resolved, report = resolve_laterality(mask, _make_affine())

        # Tumor should be remapped to unified label 5
        assert np.any(resolved == 5), "Tumor should be preserved as label 5"
        # Cyst should be remapped to unified label 6
        assert np.any(resolved == 6), "Cyst should be preserved as label 6"
        # Kidneys should still be assigned
        assert np.any(resolved == 2), "Right kidney present"
        assert np.any(resolved == 3), "Left kidney present"

    def test_background_unchanged(self):
        mask = np.zeros((64, 64, 64), dtype=np.int16)
        mask[5:15, 25:35, 25:35] = 1
        mask[45:55, 25:35, 25:35] = 1

        resolved, report = resolve_laterality(mask, _make_affine())

        # Background (label 0) should dominate
        assert (resolved == 0).sum() > (resolved > 0).sum()
