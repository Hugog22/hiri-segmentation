"""
Data augmentation transforms for 3D medical image segmentation.

Includes the critical left-right flip with kidney label swap, and
a comprehensive set of spatial and intensity augmentations.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Unified label map
LABEL_RIGHT_KIDNEY = 2
LABEL_LEFT_KIDNEY = 3
LABEL_RENAL_TUMOR = 5
LABEL_RENAL_CYST = 6

# Labels that must be swapped when flipping left-right
LR_SWAP_PAIRS = [
    (LABEL_RIGHT_KIDNEY, LABEL_LEFT_KIDNEY),
    # If we add lateralized tumors/cysts in the future, add them here
]


def augment_flip_lr(
    image: np.ndarray,
    mask: np.ndarray,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip image and mask along the left-right axis (axis 0 in RAS) and
    swap kidney laterality labels (2↔3).

    This is the ONLY flip that requires label swapping. Flips along
    AP (axis 1) and SI (axis 2) do NOT change labels.

    Args:
        image: 3D image array (X, Y, Z) in RAS orientation
        mask: 3D label array with unified label encoding
        axis: axis to flip (0 = left-right in RAS)

    Returns:
        Tuple of (flipped_image, flipped_mask_with_swapped_labels)
    """
    flipped_img = np.flip(image, axis=axis).copy()
    flipped_mask = np.flip(mask, axis=axis).copy()

    # Swap lateralized labels
    out_mask = flipped_mask.copy()
    for label_a, label_b in LR_SWAP_PAIRS:
        out_mask[flipped_mask == label_a] = label_b
        out_mask[flipped_mask == label_b] = label_a

    return flipped_img, out_mask


def augment_flip(
    image: np.ndarray,
    mask: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip image and mask along a given axis.
    Only axis=0 (left-right) triggers label swapping.
    """
    if axis == 0:
        return augment_flip_lr(image, mask, axis=0)
    else:
        return (
            np.flip(image, axis=axis).copy(),
            np.flip(mask, axis=axis).copy(),
        )


def apply_random_flips(
    image: np.ndarray,
    mask: np.ndarray,
    axes: list[int] = (0, 1, 2),
    prob: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply random flips along specified axes with given probability.

    Args:
        image: 3D image array
        mask: 3D label array
        axes: which axes to consider flipping
        prob: probability of flipping each axis
        rng: numpy random generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng()

    for axis in axes:
        if rng.random() < prob:
            image, mask = augment_flip(image, mask, axis)

    return image, mask
