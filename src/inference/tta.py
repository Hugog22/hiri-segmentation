"""
Test-Time Augmentation (TTA) module for robust inference.
"""
import logging
from typing import List, Dict, Optional

import torch
import numpy as np
from .sliding_window import sliding_window_inference_3d

logger = logging.getLogger(__name__)

def get_tta_transforms(config: dict) -> List[Dict]:
    """
    Returns list of transform specifications with flip axes and whether kidney swap is needed.
    """
    swap_dict = config.get("augmentation", {}).get("random_flip", {}).get("swap_labels_on_lr_flip", {2: 3, 3: 2})
    
    return [
        {"name": "original", "flip_dims": None, "swap_kidneys": False, "swap_dict": None},
        {"name": "flip_lr", "flip_dims": [2], "swap_kidneys": True, "swap_dict": swap_dict},
        {"name": "flip_ap", "flip_dims": [3], "swap_kidneys": False, "swap_dict": None},
        {"name": "flip_lr_ap", "flip_dims": [2, 3], "swap_kidneys": True, "swap_dict": swap_dict}
    ]

def tta_inference(
    model: torch.nn.Module, 
    image: torch.Tensor, 
    patch_size: tuple, 
    overlap: float, 
    device: torch.device, 
    tta_transforms: Optional[List[Dict]] = None
) -> np.ndarray:
    """
    Perform Test-Time Augmentation inference.
    """
    if tta_transforms is None:
        # Default config mimicking our task
        tta_transforms = get_tta_transforms({"augmentation": {"random_flip": {"swap_labels_on_lr_flip": {2: 3, 3: 2}}}})
        
    all_probs = []
    
    for tform in tta_transforms:
        logger.info(f"Running TTA transform: {tform['name']}")
        
        # 1. Apply transform to input image
        if tform["flip_dims"] is not None:
            input_tensor = torch.flip(image, dims=tform["flip_dims"])
        else:
            input_tensor = image
            
        # 2. Run sliding window inference
        # Returns numpy array of shape (1, C, D, H, W)
        probs_np = sliding_window_inference_3d(
            model=model,
            image=input_tensor,
            patch_size=patch_size,
            overlap=overlap,
            device=device
        )
        probs_tensor = torch.from_numpy(probs_np)
        
        # 3. Reverse the transform on the output softmax
        if tform["flip_dims"] is not None:
            probs_tensor = torch.flip(probs_tensor, dims=tform["flip_dims"])
            
        # 4. For L-R flip reversal: MUST swap probability channels for right/left kidney
        if tform["swap_kidneys"]:
            logger.debug(f"Swapping kidney channels for transform {tform['name']}")
            probs_corrected = probs_tensor.clone()
            
            # Using the exact swapping logic required
            probs_corrected[:, 2, ...] = probs_tensor[:, 3, ...]
            probs_corrected[:, 3, ...] = probs_tensor[:, 2, ...]
            
            probs_tensor = probs_corrected
            
        all_probs.append(probs_tensor.numpy())
        
    # Average all softmax outputs
    avg_probs = np.mean(all_probs, axis=0)
    
    return avg_probs
