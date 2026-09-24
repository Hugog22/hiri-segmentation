"""
Sliding window inference module for 3D medical image segmentation.
"""
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch
import numpy as np
import nibabel as nib
import yaml

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, Invertd
)
from monai.data import dict_to_array

logger = logging.getLogger(__name__)

def sliding_window_inference_3d(
    model: torch.nn.Module, 
    image: torch.Tensor, 
    patch_size: Tuple[int, int, int], 
    overlap: float, 
    device: torch.device, 
    mode: str = 'gaussian', 
    sigma_scale: float = 0.125, 
    sw_batch_size: int = 4, 
    progress: bool = True
) -> np.ndarray:
    """
    Wraps MONAI's sliding_window_inference for our 3D segmentation use case.
    
    Args:
        model: PyTorch model.
        image: Single 3D image tensor (1, 1, D, H, W) or (1, 1, X, Y, Z).
        patch_size: Patch size for inference.
        overlap: Overlap fraction (0.0 to 1.0).
        device: Target device (CPU/GPU).
        mode: Blending mode, e.g., 'gaussian' or 'constant'.
        sigma_scale: Sigma scale for Gaussian blending.
        sw_batch_size: Number of sliding window patches to process at once.
        progress: Whether to show progress bar.
        
    Returns:
        Softmax probability map as a numpy array (1, C, D, H, W).
    """
    model.eval()
    image = image.to(device)
    
    with torch.no_grad():
        try:
            logits = sliding_window_inference(
                inputs=image,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode=mode,
                sigma_scale=sigma_scale,
                progress=progress
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and sw_batch_size > 1:
                logger.warning(f"OOM error. Falling back to smaller batch size (sw_batch_size={sw_batch_size//2})")
                torch.cuda.empty_cache()
                return sliding_window_inference_3d(
                    model=model, image=image, patch_size=patch_size, overlap=overlap, 
                    device=device, mode=mode, sigma_scale=sigma_scale, 
                    sw_batch_size=sw_batch_size // 2, progress=progress
                )
            else:
                raise e
        
        probs = torch.softmax(logits, dim=1)
        
    return probs.cpu().numpy()

def predict_single_volume(
    model: torch.nn.Module, 
    image_path: Path, 
    config: Dict[str, Any], 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads NIfTI image, applies transforms, runs inference, and returns to ORIGINAL space.
    
    Args:
        model: The trained PyTorch model.
        image_path: Path to the input NIfTI image.
        config: Inference configuration dictionary.
        device: Target PyTorch device.
        
    Returns:
        Tuple of (softmax_probs, segmentation_argmax) in original space.
    """
    target_spacing = tuple(config.get("preprocessing", {}).get("target_spacing", [1.0, 1.0, 1.5]))
    hu_window = config.get("preprocessing", {}).get("hu_window", [-150.0, 250.0])
    
    # Define preprocessing
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=float(hu_window[0]), 
            a_max=float(hu_window[1]), 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
    ])
    
    data = {"image": str(image_path)}
    data = val_transforms(data)
    
    input_tensor = data["image"].unsqueeze(0)  # (1, 1, D, H, W)
    
    patch_size = tuple(config.get("inference", {}).get("patch_size", 
                       config.get("training", {}).get("patch_size", (128, 128, 128))))
    overlap = config.get("inference", {}).get("sliding_window_overlap", 0.5)
    sigma = config.get("inference", {}).get("gaussian_sigma", 0.125)
    
    probs = sliding_window_inference_3d(
        model=model,
        image=input_tensor,
        patch_size=patch_size,
        overlap=overlap,
        device=device,
        mode='gaussian',
        sigma_scale=sigma,
        sw_batch_size=4
    )
    
    # Store predictions back in the dict for MONAI inversion
    data["pred"] = probs[0] # (C, D, H, W)
    
    # Invert transforms to get back to original space
    inverter = Invertd(
        keys=["pred"], 
        transform=val_transforms, 
        orig_keys="image", 
        nearest_interp=False, 
        to_tensor=True
    )
    inverted_data = inverter(data)
    
    original_probs = inverted_data["pred"].numpy()
    
    # Add batch dim back if needed or keep it (C, D, H, W)
    segmentation_argmax = np.argmax(original_probs, axis=0)
    
    return np.expand_dims(original_probs, axis=0), segmentation_argmax

def predict_batch(
    model: torch.nn.Module, 
    image_paths: List[Path], 
    config: Dict[str, Any], 
    device: torch.device, 
    output_dir: Path
) -> List[Path]:
    """
    Runs prediction on multiple volumes and saves each segmentation as NIfTI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    
    for img_path in image_paths:
        logger.info(f"Predicting volume: {img_path.name}")
        _, seg_mask = predict_single_volume(model, img_path, config, device)
        
        # Load original image to get header and affine
        orig_img = nib.load(str(img_path))
        
        out_nifti = nib.Nifti1Image(seg_mask.astype(np.uint8), orig_img.affine, orig_img.header)
        out_path = output_dir / f"{img_path.name.replace('.nii.gz', '')}_seg.nii.gz"
        nib.save(out_nifti, str(out_path))
        output_paths.append(out_path)
        
        logger.info(f"Saved segmentation to {out_path}")
        
    return output_paths

if __name__ == "__main__":
    pass
