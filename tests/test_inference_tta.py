"""
Tests for Test-Time Augmentation (TTA) transforms and kidney channel swap logic.
"""
import numpy as np
import pytest
import torch

from src.inference.tta import get_tta_transforms, tta_inference


class DummyModel(torch.nn.Module):
    """
    A deterministic dummy model that predicts a localized right kidney (class 2)
    and left kidney (class 3).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is (B, 1, D, H, W)
        B, _, D, H, W = x.shape
        logits = torch.zeros((B, 4, D, H, W), dtype=torch.float32)

        # Place class 2 (Right Kidney) at lower spatial dim 0 (D)
        # Place class 3 (Left Kidney) at higher spatial dim 0 (D)
        d_mid = D // 2
        logits[:, 2, :d_mid, :, :] = 5.0
        logits[:, 3, d_mid:, :, :] = 5.0
        return logits


class TestTTATransforms:
    def test_get_tta_transforms_structure(self):
        config = {
            "augmentation": {
                "random_flip": {
                    "swap_labels_on_lr_flip": {2: 3, 3: 2}
                }
            }
        }
        tforms = get_tta_transforms(config)
        assert len(tforms) == 4
        names = [t["name"] for t in tforms]
        assert "original" in names
        assert "flip_lr" in names
        assert "flip_ap" in names
        assert "flip_lr_ap" in names

        # flip_lr must flag swap_kidneys=True
        lr_tform = next(t for t in tforms if t["name"] == "flip_lr")
        assert lr_tform["swap_kidneys"] is True
        assert lr_tform["flip_dims"] == [2]

    def test_tta_inference_preserves_shape(self):
        model = DummyModel()
        image = torch.zeros((1, 1, 16, 16, 16))
        patch_size = (16, 16, 16)
        overlap = 0.5
        device = torch.device("cpu")

        probs = tta_inference(
            model=model,
            image=image,
            patch_size=patch_size,
            overlap=overlap,
            device=device
        )

        assert probs.shape == (1, 4, 16, 16, 16)
        # Check that probabilities sum to 1 along channel dimension
        prob_sums = np.sum(probs, axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-4)
