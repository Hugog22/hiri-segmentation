import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss as MonaiDiceCELoss
from scipy.ndimage import distance_transform_edt


class DiceCELoss(nn.Module):
    """
    Combination of Dice loss and Cross-Entropy loss.
    Wraps MONAI's implementation for our specific requirements.
    """
    def __init__(self, lambda_dice: float = 1.0, lambda_ce: float = 1.0):
        super().__init__()
        self.loss_fn = MonaiDiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce
        )

    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class BoundaryLoss(nn.Module):
    """
    Boundary loss based on distance transforms (Kervadec et al., 2019).
    Useful for fine boundary refinement.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

    def compute_dtm(self, label_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute distance transform map for each foreground class in label_batch.
        Returns:
            Tensor of shape (B, num_classes - 1, D, H, W)
        """
        b_size = label_batch.shape[0]
        spatial_shape = label_batch.shape[2:]
        dtm_np = np.zeros((b_size, self.num_classes - 1, *spatial_shape), dtype=np.float32)
        
        label_np = label_batch.detach().cpu().numpy()
        
        for b in range(b_size):
            for c_idx, c in enumerate(range(1, self.num_classes)):
                posmask = (label_np[b, 0] == c)
                if not posmask.any():
                    continue
                negmask = ~posmask
                
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                dtm_np[b, c_idx] = negdis - posdis
                
        return torch.from_numpy(dtm_np).to(label_batch.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred shape: (B, C, D, H, W)
        # target shape: (B, 1, D, H, W)
        probs = F.softmax(pred, dim=1)
        dtm = self.compute_dtm(target)  # (B, C-1, D, H, W)
        
        # Multiply foreground class probabilities with corresponding DTM
        fg_probs = probs[:, 1:self.num_classes]  # (B, C-1, D, H, W)
        loss = torch.mean(fg_probs * dtm)
        return loss


class DeepSupervisionLoss(nn.Module):
    """
    Computes a weighted sum of losses at different scales.
    """
    def __init__(self, base_loss_fn: nn.Module, weights: list = [1.0, 0.5, 0.25, 0.125]):
        super().__init__()
        self.base_loss = base_loss_fn
        self.weights = weights

    def forward(self, preds, target):
        if not isinstance(preds, (list, tuple)):
            # Fallback if model didn't return a list
            return self.base_loss(preds, target)
            
        loss = 0.0
        for i, pred in enumerate(preds):
            if i >= len(self.weights):
                break
                
            weight = self.weights[i]
            if weight == 0:
                continue
                
            # Downsample target to match pred spatial dimensions
            if pred.shape[2:] != target.shape[2:]:
                target_down = F.interpolate(
                    target.float(), 
                    size=pred.shape[2:], 
                    mode="nearest"
                ).long()
            else:
                target_down = target
                
            loss += weight * self.base_loss(pred, target_down)
            
        return loss


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function for loss functions.
    """
    loss_type = config.get("training", {}).get("loss", "dice_ce")
    
    if loss_type == "dice_ce":
        base_loss = DiceCELoss()
    else:
        # Default fallback
        base_loss = DiceCELoss()
        
    use_deep_supervision = config.get("training", {}).get("deep_supervision", False)
    
    if use_deep_supervision:
        return DeepSupervisionLoss(base_loss)
    return base_loss
