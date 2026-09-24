import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.utils.data import DistributedSampler

logger = logging.getLogger(__name__)


class LabelSwapTransform(MapTransform):
    """
    Custom transform to swap right (2) and left (3) kidneys when an L-R flip is applied.
    """
    def __init__(self, keys: List[str], swap_prob: float = 0.5):
        super().__init__(keys)
        self.swap_prob = swap_prob
        self.flipper = RandFlipd(keys=keys, prob=1.0, spatial_axis=0)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if np.random.rand() < self.swap_prob:
            # Apply L-R flip
            d = self.flipper(d)
            # Swap labels 2 and 3 if 'label' is in keys
            if "label" in self.keys and "label" in d:
                lbl = d["label"]
                if isinstance(lbl, torch.Tensor):
                    temp_lbl = lbl.clone()
                    temp_lbl[lbl == 2] = 3
                    temp_lbl[lbl == 3] = 2
                    d["label"] = temp_lbl
                else:
                    temp_lbl = np.copy(lbl)
                    temp_lbl[lbl == 2] = 3
                    temp_lbl[lbl == 3] = 2
                    d["label"] = temp_lbl
        return d


def get_train_transforms(config: dict) -> Compose:
    """
    Returns the MONAI compose pipeline for training transforms.
    """
    target_spacing = config.get("preprocessing", {}).get("target_spacing", [1.0, 1.0, 1.5])
    hu_window = config.get("preprocessing", {}).get("hu_window", [-150, 250])
    patch_size = config.get("training", {}).get("patch_size", [128, 128, 128])

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], 
            pixdim=target_spacing, 
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=hu_window[0], 
            a_max=hu_window[1], 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        # Augmentations
        # Custom L-R flip with label swap
        LabelSwapTransform(keys=["image", "label"], swap_prob=0.5),
        # Flip along axis 1 (A-P) without swapping labels
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        # Flip along axis 2 (S-I) without swapping labels
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
        # Rotate 90 degrees only on axes (1,2) to avoid L-R confusion
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),  # ±15°
            scale_range=(0.15, 0.15, 0.15),  # [0.85, 1.15] (close to [0.85, 1.25])
        ),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.0)),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.7, 1.5)),
    ])


def get_val_transforms(config: dict) -> Compose:
    """
    Returns the MONAI compose pipeline for validation transforms.
    """
    target_spacing = config.get("preprocessing", {}).get("target_spacing", [1.0, 1.0, 1.5])
    hu_window = config.get("preprocessing", {}).get("hu_window", [-150, 250])

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], 
            pixdim=target_spacing, 
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=hu_window[0], 
            a_max=hu_window[1], 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])


def resolve_case_paths(case_id: str, data_dir: Path) -> Dict[str, str]:
    """
    Resolves the actual image and label file paths for a given case ID.
    case_id format: "{dataset_name}_{case_subid}"
    Example: totalseg_s0001
    """
    dataset_name = case_id.split("_")[0]
    case_subid = case_id[len(dataset_name)+1:]
    
    # We construct the path directly based on the expected structure
    case_dir = data_dir / dataset_name / case_id
    image_path = case_dir / "image.nii.gz"
    label_path = case_dir / "unified_label.nii.gz"
    
    if not image_path.exists() or not label_path.exists():
        # Sometimes case_id might be the exact folder name
        case_dir = data_dir / dataset_name / case_subid
        image_path = case_dir / "image.nii.gz"
        label_path = case_dir / "unified_label.nii.gz"
        
    return {
        "image": str(image_path),
        "label": str(label_path)
    }


def get_dataloaders(
    config: dict, 
    fold: int, 
    splits_path: str, 
    data_dir: str, 
    is_ddp: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation dataloaders based on the splits file.
    """
    splits_file = Path(splits_path)
    data_path = Path(data_dir)
    
    with open(splits_file, "r") as f:
        splits = json.load(f)
        
    fold_data = splits["folds"][str(fold)]
    train_cases = fold_data["train"]
    val_cases = fold_data["val"]
    
    train_files = [resolve_case_paths(case, data_path) for case in train_cases]
    val_files = [resolve_case_paths(case, data_path) for case in val_cases]
    
    # Filter out missing files
    train_files = [f for f in train_files if Path(f["image"]).exists() and Path(f["label"]).exists()]
    val_files = [f for f in val_files if Path(f["image"]).exists() and Path(f["label"]).exists()]
    
    logger.info(f"Found {len(train_files)} training cases and {len(val_files)} validation cases for fold {fold}")

    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)

    # Use CacheDataset for better performance if possible, else fallback to Dataset
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

    batch_size = config.get("training", {}).get("batch_size", 2)
    cpus = config.get("training", {}).get("num_workers", 4)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=cpus, 
        pin_memory=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1,  # Val uses batch size 1 for sliding window
        shuffle=False, 
        num_workers=cpus, 
        pin_memory=True,
        sampler=val_sampler
    )

    return train_loader, val_loader
