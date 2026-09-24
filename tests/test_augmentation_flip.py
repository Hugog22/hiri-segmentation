"""
Tests for the left-right flip augmentation with kidney label swapping.

Critical: flipping along the L-R axis (axis 0 in RAS) must simultaneously
swap right kidney (label 2) ↔ left kidney (label 3). Flips along other
axes must NOT change labels.
"""
import numpy as np
import pytest

from src.training.augmentations import augment_flip_lr, augment_flip


class TestLRFlipSwapsKidneys:
    """Left-right flip must swap kidney labels 2↔3."""

    def test_basic_swap(self):
        img = np.random.rand(20, 20, 20)
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[3, 10, 10] = 2  # right kidney at low X
        mask[17, 10, 10] = 3  # left kidney at high X

        f_img, f_mask = augment_flip_lr(img, mask)

        # After flip: voxel at X=3 moves to X=16, voxel at X=17 moves to X=2
        # AND labels are swapped: 2→3, 3→2
        assert f_mask[16, 10, 10] == 3, "Right kidney (2) → flipped position → should become left (3)"
        assert f_mask[2, 10, 10] == 2, "Left kidney (3) → flipped position → should become right (2)"

    def test_volumetric_swap(self):
        """Check with larger regions, not just single voxels."""
        mask = np.zeros((40, 40, 40), dtype=np.uint8)
        mask[5:15, 15:25, 15:25] = 2   # right kidney block
        mask[25:35, 15:25, 15:25] = 3  # left kidney block

        _, f_mask = augment_flip_lr(np.zeros_like(mask, dtype=float), mask)

        # Right kidney count should be preserved
        assert np.sum(mask == 2) == np.sum(f_mask == 2), "Right kidney voxel count preserved"
        assert np.sum(mask == 3) == np.sum(f_mask == 3), "Left kidney voxel count preserved"


class TestLRFlipPreservesLiver:
    """Liver (label 1) should not change on L-R flip."""

    def test_liver_unchanged(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1  # liver

        _, f_mask = augment_flip_lr(np.zeros_like(mask, dtype=float), mask)

        assert np.sum(f_mask == 1) == np.sum(mask == 1), "Liver voxel count unchanged"
        assert not np.any(f_mask == 2), "No spurious right kidney"
        assert not np.any(f_mask == 3), "No spurious left kidney"


class TestLRFlipPreservesBackground:
    """Background should be unchanged."""

    def test_all_background(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        _, f_mask = augment_flip_lr(np.zeros_like(mask, dtype=float), mask)
        assert np.all(f_mask == 0)


class TestDoubleFlipIdentity:
    """Flipping L-R twice should return to the original."""

    def test_image_identity(self):
        rng = np.random.default_rng(42)
        img = rng.standard_normal((15, 15, 15))
        mask = rng.integers(0, 4, size=(15, 15, 15), dtype=np.uint8)

        f1_img, f1_mask = augment_flip_lr(img, mask)
        f2_img, f2_mask = augment_flip_lr(f1_img, f1_mask)

        np.testing.assert_allclose(f2_img, img)
        np.testing.assert_array_equal(f2_mask, mask)


class TestAPFlipNoLabelChange:
    """Flipping along axis 1 (anterior-posterior) must NOT swap labels."""

    def test_ap_flip(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5, 3, 10] = 2
        mask[15, 17, 10] = 3

        _, f_mask = augment_flip(np.zeros_like(mask, dtype=float), mask, axis=1)

        # Labels should be the same — only spatial position changes
        unique_before = set(np.unique(mask))
        unique_after = set(np.unique(f_mask))
        assert unique_before == unique_after, "Same labels present"
        assert np.sum(f_mask == 2) == np.sum(mask == 2), "Label 2 count unchanged"
        assert np.sum(f_mask == 3) == np.sum(mask == 3), "Label 3 count unchanged"


class TestSIFlipNoLabelChange:
    """Flipping along axis 2 (superior-inferior) must NOT swap labels."""

    def test_si_flip(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5, 10, 3] = 2
        mask[15, 10, 17] = 3

        _, f_mask = augment_flip(np.zeros_like(mask, dtype=float), mask, axis=2)

        assert np.sum(f_mask == 2) == np.sum(mask == 2)
        assert np.sum(f_mask == 3) == np.sum(mask == 3)


class TestFlipWithTumors:
    """Tumor and cyst labels adjacent to kidneys should follow the spatial flip
    but NOT be swapped (they are label 4/5/6, not 2/3)."""

    def test_tumor_follows_flip_no_label_change(self):
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        # Right kidney with embedded hepatic tumor (label 4 = hepatic tumor)
        mask[5:10, 10:15, 10:15] = 2
        mask[6:8, 11:13, 11:13] = 4  # tumor inside liver area

        _, f_mask = augment_flip_lr(np.zeros_like(mask, dtype=float), mask)

        # Hepatic tumor (4) should NOT change label — it's not lateralized
        assert np.sum(f_mask == 4) == np.sum(mask == 4), "Hepatic tumor count unchanged"
        # Kidney label should swap
        assert np.sum(f_mask == 3) == np.sum(mask == 2), "Right kidney became left after flip"

    def test_renal_tumor_preserved(self):
        """Renal tumor (label 5) is NOT currently lateralized, so no swap."""
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[5:10, 10:15, 10:15] = 2   # right kidney
        mask[20:25, 10:15, 10:15] = 3  # left kidney
        mask[6:8, 11:13, 11:13] = 5    # renal tumor

        _, f_mask = augment_flip_lr(np.zeros_like(mask, dtype=float), mask)

        assert np.sum(f_mask == 5) == np.sum(mask == 5), "Renal tumor count unchanged"

    def test_all_labels_present_after_flip(self):
        """Comprehensive test with all labels."""
        mask = np.zeros((40, 40, 40), dtype=np.uint8)
        mask[5:10, 15:20, 15:20] = 1   # liver
        mask[10:15, 15:20, 15:20] = 2  # right kidney
        mask[25:30, 15:20, 15:20] = 3  # left kidney
        mask[6:8, 16:18, 16:18] = 4    # hepatic tumor
        mask[11:13, 16:18, 16:18] = 5  # renal tumor
        mask[26:28, 16:18, 16:18] = 6  # renal cyst

        _, f_mask = augment_flip_lr(np.zeros_like(mask, dtype=float), mask)

        for label_val in [0, 1, 2, 3, 4, 5, 6]:
            assert np.any(f_mask == label_val) == np.any(mask == label_val), \
                f"Label {label_val} presence should be consistent"
