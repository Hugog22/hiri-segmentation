"""
Module for exporting segmentations to STL meshes for 3D printing.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import nibabel as nib
import trimesh
from skimage.measure import marching_cubes
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)

def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """
    Reduce mesh complexity for 3D printing.
    Note: Requires quadratic edge collapse or open3d/pymeshlab, 
    but we will try a basic strategy if possible.
    """
    if len(mesh.faces) <= target_faces:
        return mesh
        
    logger.info(f"Simplifying mesh from {len(mesh.faces)} to {target_faces} faces.")
    # trimesh simplification might be limited depending on the environment, 
    # but we can call decimate if available (e.g. if pyembree/fast_simplification is installed)
    try:
        import fast_simplification
        vertices, faces = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=target_faces
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    except ImportError:
        logger.warning("fast_simplification not installed, skipping mesh simplification.")
        return mesh

def segmentation_to_stl(
    mask_path: Path, 
    output_dir: Path, 
    label_map: Dict[int, str], 
    smoothing: float = 0.5,
    simplify_faces: int = 100000
) -> List[Path]:
    """
    Convert a labeled segmentation NIfTI to separate STL meshes per label.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    spacing = tuple(img.header.get_zooms()[:3])
    
    for label_idx, label_name in label_map.items():
        binary_mask = (data == label_idx).astype(np.float32)
        
        if not np.any(binary_mask):
            logger.info(f"Label {label_name} ({label_idx}) not found in mask, skipping.")
            continue
            
        logger.info(f"Processing {label_name} for STL export...")
        
        # Apply Gaussian smoothing
        if smoothing > 0:
            binary_mask = ndimage.gaussian_filter(binary_mask, sigma=smoothing)
            
        # Run marching cubes
        verts, faces, normals, values = marching_cubes(binary_mask, level=0.5, spacing=spacing)
        
        # Build mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Simplify if requested
        if simplify_faces > 0:
            mesh = simplify_mesh(mesh, simplify_faces)
            
        out_stl = output_dir / f"{label_name}.stl"
        mesh.export(str(out_stl))
        logger.info(f"Saved {out_stl}")
        generated_files.append(out_stl)
        
    return generated_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export segmentations to STL.")
    parser.add_argument("--mask-path", type=Path, required=True, help="Input NIfTI mask")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--smoothing", type=float, default=0.5, help="Gaussian smoothing sigma")
    parser.add_argument("--simplify", type=int, default=100000, help="Target faces for simplification")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    # default label map
    label_map = {1: "liver", 2: "right_kidney", 3: "left_kidney"}
    segmentation_to_stl(args.mask_path, args.output_dir, label_map, args.smoothing, args.simplify)
