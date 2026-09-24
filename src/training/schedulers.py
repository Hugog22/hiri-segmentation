import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    PolynomialLR,
    _LRScheduler
)


class WarmupScheduler(_LRScheduler):
    """
    Wraps another scheduler with a linear warmup phase.
    """
    def __init__(self, optimizer, base_scheduler, warmup_epochs, last_epoch=-1):
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step(epoch)
        super().step(epoch)


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> _LRScheduler:
    """
    Factory function for learning rate schedulers.
    """
    scheduler_type = config.get("training", {}).get("scheduler", "CosineAnnealingLR")
    epochs = config.get("training", {}).get("epochs", {}).get("monai", 500)
    
    if scheduler_type == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    elif scheduler_type == "PolynomialLR" or scheduler_type == "poly":
        return PolynomialLR(optimizer, total_iters=epochs, power=0.9)
        
    elif scheduler_type == "warmup_cosine":
        base_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=epochs - 50, 
            eta_min=1e-7
        )
        return WarmupScheduler(optimizer, base_scheduler, warmup_epochs=50)
        
    elif scheduler_type == "CosineAnnealingWarmRestarts" or scheduler_type == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
    else:
        # Default to CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
