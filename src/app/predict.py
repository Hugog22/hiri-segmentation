"""
End-to-end prediction script.
"""
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import numpy as np
import nibabel as nib

# Assuming src is in PYTHONPATH
from src.inference.sliding_window import predict_single_volume, sliding_window_inference_3d
from src.inference.tta import tta_inference, get_tta_transforms
from src.inference.postprocessing import postprocess_segmentation, compute_component_volumes
from src.app.export_stl import segmentation_to_stl
from src.app.report import generate_volume_report

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, Invertd
)
from monai.networks.nets import SwinUNETR # Example model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    from src.training.models import create_model

    target_spacing = tuple(config.get("preprocessing", {}).get("target_spacing", [1.0, 1.0, 1.5]))
    hu_window = config.get("preprocessing", {}).get("hu_window", [-150.0, 250.0])

    # 1. Load Input and 2. Preprocess
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
    
    logger.info(f"Processing input file: {args.input}")
    data = {"image": str(args.input)}
    data = val_transforms(data)
    input_tensor = data["image"].unsqueeze(0).to(device)
    
    logger.info(f"Creating model '{args.model_name}' and loading weights from {args.model_path}")
    model = create_model(args.model_name, config).to(device)
    checkpoint = torch.load(str(args.model_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    # Strip potential 'module.' prefix from DDP
    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    
    patch_size = tuple(config.get("inference", {}).get("patch_size", 
                       config.get("training", {}).get("patch_size", (128, 128, 128))))
    overlap = config.get("inference", {}).get("sliding_window_overlap", 0.5)
    
    # 3. Run model inference
    if args.tta:
        logger.info("Running inference with TTA")
        probs = tta_inference(model, input_tensor, patch_size, overlap, device)
    else:
        logger.info("Running standard inference")
        probs = sliding_window_inference_3d(
            model=model, image=input_tensor, patch_size=patch_size, 
            overlap=overlap, device=device
        )
        
    data["pred"] = probs[0] # (C, D, H, W)
    
    # Reverse transforms
    inverter = Invertd(
        keys=["pred"], transform=val_transforms, orig_keys="image", 
        nearest_interp=False, to_tensor=True
    )
    inverted_data = inverter(data)
    original_probs = inverted_data["pred"].numpy()
    segmentation_argmax = np.argmax(original_probs, axis=0)
    
    # 4. Postprocess
    orig_img = nib.load(str(args.input))
    spacing = tuple(orig_img.header.get_zooms()[:3])
    logger.info("Running postprocessing")
    final_seg = postprocess_segmentation(segmentation_argmax, spacing, config)
    
    # 5. Save segmentation
    seg_path = args.output_dir / "segmentation.nii.gz"
    out_nifti = nib.Nifti1Image(final_seg.astype(np.uint8), orig_img.affine, orig_img.header)
    nib.save(out_nifti, str(seg_path))
    logger.info(f"Saved segmentation to {seg_path}")
    
    # 6. Compute volumes
    volumes = {}
    volumes["liver_ml"] = compute_component_volumes((final_seg == 1).astype(np.uint8), spacing)[0] if np.any(final_seg == 1) else 0.0
    volumes["right_kidney_ml"] = compute_component_volumes((final_seg == 2).astype(np.uint8), spacing)[0] if np.any(final_seg == 2) else 0.0
    volumes["left_kidney_ml"] = compute_component_volumes((final_seg == 3).astype(np.uint8), spacing)[0] if np.any(final_seg == 3) else 0.0
    
    with open(args.output_dir / "volumes.json", "w") as f:
        json.dump(volumes, f, indent=4)
        
    # 7. Generate report
    generate_volume_report(volumes, args.output_dir / "report.txt")
    
    # Optional STL export
    if args.export_stl:
        logger.info("Exporting STL meshes")
        label_map = {1: "liver", 2: "right_kidney", 3: "left_kidney"}
        segmentation_to_stl(seg_path, args.output_dir, label_map)
        
    logger.info("Pipeline completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="End-to-end medical segmentation pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="Input NIfTI or DICOM path")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to weights")
    parser.add_argument("--model-name", type=str, required=True, help="Model arch name")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--export-stl", action="store_true", help="Export STL meshes")
    
    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
