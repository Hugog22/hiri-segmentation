import logging
import urllib.request
from typing import List, Optional

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR, UNETR

logger = logging.getLogger(__name__)


def create_model(model_name: str, config: dict) -> nn.Module:
    """
    Factory function to create the segmentation model.
    """
    patch_size = config.get("training", {}).get("patch_size", [128, 128, 128])
    out_channels = 4  # Background, Liver, Right Kidney, Left Kidney
    
    if model_name == "swin_unetr_base":
        model = SwinUNETR(
            img_size=patch_size,
            in_channels=1,
            out_channels=out_channels,
            feature_size=48,
            use_v2=True
        )
    elif model_name == "swin_unetr_large":
        model = SwinUNETR(
            img_size=patch_size,
            in_channels=1,
            out_channels=out_channels,
            feature_size=96,
            use_v2=True
        )
    elif model_name == "unetr":
        model = UNETR(
            img_size=patch_size,
            in_channels=1,
            out_channels=out_channels,
            feature_size=64
        )
    elif model_name == "segresnet":
        model = SegResNet(
            spatial_dims=3,
            init_filters=32,
            in_channels=1,
            out_channels=out_channels,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    param_count = get_model_param_count(model)
    logger.info(f"Created model {model_name} with {param_count:,} parameters.")
    
    return model


def load_pretrained_swin_unetr(model: SwinUNETR, weights_path_or_url: str):
    """
    Loads pretrained SSL weights into the SwinUNETR encoder.
    """
    try:
        if weights_path_or_url.startswith("http"):
            logger.info(f"Downloading pretrained weights from {weights_path_or_url}")
            weights_path = "/tmp/swin_unetr_pretrained.pth"
            urllib.request.urlretrieve(weights_path_or_url, weights_path)
        else:
            weights_path = weights_path_or_url
            
        logger.info(f"Loading pretrained weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        # Filter only encoder weights
        encoder_weights = {k: v for k, v in state_dict.items() if "encoder" in k}
        
        # Load weights
        model.load_state_dict(encoder_weights, strict=False)
        logger.info("Successfully loaded pretrained encoder weights.")
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")


def get_model_param_count(model: nn.Module) -> int:
    """
    Returns the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DeepSupervisionWrapper(nn.Module):
    """
    Wrapper for models that output multiple scales for deep supervision.
    """
    def __init__(self, model: nn.Module, weights: Optional[List[float]] = None):
        super().__init__()
        self.model = model
        self.weights = weights or [1.0, 0.5, 0.25, 0.125]

    def forward(self, x):
        return self.model(x)
