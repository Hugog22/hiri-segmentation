"""
Tests for loss functions: DiceCELoss, BoundaryLoss, and DeepSupervisionLoss.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.losses import BoundaryLoss, DeepSupervisionLoss, DiceCELoss


class TestDiceCELoss:
    def test_perfect_prediction_low_loss(self):
        loss_fn = DiceCELoss()
        # Batch size 2, 4 classes, 16x16x16
        target = torch.randint(0, 4, (2, 1, 16, 16, 16)).long()
        # Create logits that strongly predict the target
        pred = torch.zeros((2, 4, 16, 16, 16), dtype=torch.float32)
        for b in range(2):
            for c in range(4):
                pred[b, c][target[b, 0] == c] = 10.0
                pred[b, c][target[b, 0] != c] = -10.0

        loss = loss_fn(pred, target)
        assert loss.item() < 0.2, f"Loss should be near 0 for near-perfect prediction, got {loss.item()}"

    def test_output_scalar(self):
        loss_fn = DiceCELoss()
        pred = torch.randn(2, 4, 8, 8, 8)
        target = torch.randint(0, 4, (2, 1, 8, 8, 8)).long()
        loss = loss_fn(pred, target)
        assert loss.ndim == 0


class TestBoundaryLoss:
    def test_boundary_loss_multichannel_computation(self):
        loss_fn = BoundaryLoss(num_classes=4)
        target = torch.zeros((1, 1, 16, 16, 16), dtype=torch.int64)
        # Put organ 1, 2, 3 in distinct locations
        target[0, 0, 2:5, 2:5, 2:5] = 1
        target[0, 0, 7:10, 7:10, 7:10] = 2
        target[0, 0, 12:15, 12:15, 12:15] = 3

        dtm = loss_fn.compute_dtm(target)
        # Check shape: (B, num_classes-1, D, H, W)
        assert dtm.shape == (1, 3, 16, 16, 16), f"Expected (1, 3, 16, 16, 16), got {dtm.shape}"

        # Inside organ 1, channel 0 of dtm should be negative
        assert dtm[0, 0, 3, 3, 3] < 0, "Inside target organ, DTM should be negative"
        # Outside organ 1, channel 0 of dtm should be positive
        assert dtm[0, 0, 12, 12, 12] > 0, "Outside target organ, DTM should be positive"

    def test_forward_pass(self):
        loss_fn = BoundaryLoss(num_classes=4)
        target = torch.zeros((1, 1, 16, 16, 16), dtype=torch.int64)
        target[0, 0, 2:6, 2:6, 2:6] = 1
        pred = torch.randn((1, 4, 16, 16, 16))
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


class TestDeepSupervisionLoss:
    def test_multiscale_aggregation(self):
        base_loss = nn.L1Loss()
        weights = [1.0, 0.5, 0.25]
        ds_loss = DeepSupervisionLoss(base_loss, weights=weights)

        # 3 predictions at different scales
        p0 = torch.ones((1, 1, 16, 16, 16)) * 2.0
        p1 = torch.ones((1, 1, 8, 8, 8)) * 3.0
        p2 = torch.ones((1, 1, 4, 4, 4)) * 4.0
        preds = [p0, p1, p2]

        target = torch.ones((1, 1, 16, 16, 16))  # all 1.0

        # Loss 0 = |2 - 1| * 1.0 = 1.0
        # Loss 1 = |3 - 1| * 0.5 = 1.0
        # Loss 2 = |4 - 1| * 0.25 = 0.75
        # Total = 2.75
        loss = ds_loss(preds, target)
        assert np.isclose(loss.item(), 2.75, atol=1e-3)
