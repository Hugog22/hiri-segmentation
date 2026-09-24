"""
Module to resolve kidney laterality in KiTS23.

KiTS23 labels both kidneys with the same value (1). This module assigns
laterality (right=2, left=3) based on 3D connected component analysis
and centroid position in world coordinates (RAS convention).

In RAS orientation:
  - Patient's RIGHT = lower (more negative) X coordinate
  - Patient's LEFT  = higher (more positive) X coordinate
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as scipy_label
from scipy.ndimage import center_of_mass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Labels in KiTS23 original encoding
KITS_KIDNEY_LABEL = 1
KITS_TUMOR_LABEL = 2
KITS_CYST_LABEL = 3

# Our unified labels
UNIFIED_RIGHT_KIDNEY = 2
UNIFIED_LEFT_KIDNEY = 3
UNIFIED_RENAL_TUMOR = 5
UNIFIED_RENAL_CYST = 6

# Minimum volume in voxels to be considered a "real" kidney component
# (small fragments below this are merged into the nearest large component)
MIN_KIDNEY_COMPONENT_VOXELS = 500


def voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert voxel coordinates (i,j,k) to world coordinates (x,y,z) using the affine."""
    homogeneous = np.array([*voxel_coords, 1.0])
    world = affine @ homogeneous
    return world[:3]


def resolve_laterality(
    mask_data: np.ndarray,
    affine: np.ndarray,
    min_component_voxels: int = MIN_KIDNEY_COMPONENT_VOXELS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Resolve kidney laterality from a KiTS23-style mask where both kidneys
    have the same label value (1).

    Args:
        mask_data: 3D numpy array with KiTS23 label encoding
                   (1=kidney, 2=tumor, 3=cyst)
        affine: 4x4 affine matrix from the NIfTI header
        min_component_voxels: minimum volume to be considered a real component

    Returns:
        resolved_mask: 3D array with unified labels
                       (2=right_kidney, 3=left_kidney, 5=tumor, 6=cyst)
        report: dict with diagnostic information
    """
    resolved_mask = np.zeros_like(mask_data, dtype=np.uint8)

    # First, remap tumor and cyst to unified labels
    resolved_mask[mask_data == KITS_TUMOR_LABEL] = UNIFIED_RENAL_TUMOR
    resolved_mask[mask_data == KITS_CYST_LABEL] = UNIFIED_RENAL_CYST

    # Extract kidney-only binary mask
    kidney_binary = (mask_data == KITS_KIDNEY_LABEL).astype(np.uint8)

    report: dict[str, Any] = {
        'num_kidney_voxels': int(kidney_binary.sum()),
        'num_components_raw': 0,
        'num_components_major': 0,
        'right_centroid_world': None,
        'left_centroid_world': None,
        'right_volume_voxels': 0,
        'left_volume_voxels': 0,
        'confidence': 'high',
        'notes': '',
    }

    if kidney_binary.sum() == 0:
        report['confidence'] = 'error'
        report['notes'] = 'No kidney voxels found (label=1 absent)'
        logger.warning("No kidney voxels found in mask")
        return resolved_mask, report

    # 3D connected component analysis
    labeled_array, num_features = scipy_label(kidney_binary)
    report['num_components_raw'] = num_features

    if num_features == 0:
        report['confidence'] = 'error'
        report['notes'] = 'Connected component analysis found 0 components'
        return resolved_mask, report

    # Compute volume and centroid for each component
    component_info = []
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled_array == comp_id)
        volume = int(comp_mask.sum())
        centroid_voxel = np.array(center_of_mass(comp_mask))
        centroid_world = voxel_to_world(centroid_voxel, affine)
        component_info.append({
            'id': comp_id,
            'volume': volume,
            'centroid_voxel': centroid_voxel,
            'centroid_world': centroid_world,
        })

    # Separate into major and minor components
    major = [c for c in component_info if c['volume'] >= min_component_voxels]
    minor = [c for c in component_info if c['volume'] < min_component_voxels]
    report['num_components_major'] = len(major)

    if len(major) == 0:
        # All components are tiny — treat the largest one as the only kidney
        major = [max(component_info, key=lambda c: c['volume'])]
        minor = [c for c in component_info if c['id'] != major[0]['id']]
        report['confidence'] = 'low'
        report['notes'] = 'All components below minimum volume threshold'

    if len(major) == 1:
        # Single kidney (could be: solitary kidney, horseshoe, or one side
        # not captured). Assign based on position relative to midline.
        sole = major[0]
        x_world = sole['centroid_world'][0]  # X in RAS

        # Check if it spans both sides (horseshoe-like)
        comp_mask = (labeled_array == sole['id'])
        voxel_coords = np.argwhere(comp_mask)
        world_coords_x = []
        for v in voxel_coords[::100]:  # subsample for speed
            w = voxel_to_world(v.astype(float), affine)
            world_coords_x.append(w[0])
        x_range = max(world_coords_x) - min(world_coords_x) if world_coords_x else 0

        if x_range > 100:  # >100mm span suggests horseshoe kidney
            # Split by X midpoint
            x_mid = (max(world_coords_x) + min(world_coords_x)) / 2
            for v_idx in np.argwhere(comp_mask):
                w = voxel_to_world(v_idx.astype(float), affine)
                if w[0] < x_mid:
                    resolved_mask[tuple(v_idx)] = UNIFIED_RIGHT_KIDNEY
                else:
                    resolved_mask[tuple(v_idx)] = UNIFIED_LEFT_KIDNEY
            report['confidence'] = 'low'
            report['notes'] = f'Horseshoe kidney detected (X span={x_range:.1f}mm), split at midpoint'
        else:
            # True solitary kidney — assign by position
            if x_world < 0:  # Negative X in RAS = patient's right
                resolved_mask[labeled_array == sole['id']] = UNIFIED_RIGHT_KIDNEY
                report['notes'] = 'Single kidney assigned to RIGHT based on position'
            else:
                resolved_mask[labeled_array == sole['id']] = UNIFIED_LEFT_KIDNEY
                report['notes'] = 'Single kidney assigned to LEFT based on position'
            report['confidence'] = 'low'
            report['right_centroid_world'] = sole['centroid_world'].tolist() if x_world < 0 else None
            report['left_centroid_world'] = sole['centroid_world'].tolist() if x_world >= 0 else None

        # Merge minor components into the sole kidney
        for m in minor:
            resolved_mask[labeled_array == m['id']] = resolved_mask[
                labeled_array == sole['id']
            ].max()  # same label as sole

    elif len(major) >= 2:
        # Normal case: two (or more) major components
        # Sort by X world coordinate: lower X = right kidney
        major_sorted = sorted(major, key=lambda c: c['centroid_world'][0])

        right_comp = major_sorted[0]   # lowest X = patient right
        left_comp = major_sorted[-1]   # highest X = patient left

        resolved_mask[labeled_array == right_comp['id']] = UNIFIED_RIGHT_KIDNEY
        resolved_mask[labeled_array == left_comp['id']] = UNIFIED_LEFT_KIDNEY

        report['right_centroid_world'] = right_comp['centroid_world'].tolist()
        report['left_centroid_world'] = left_comp['centroid_world'].tolist()
        report['right_volume_voxels'] = right_comp['volume']
        report['left_volume_voxels'] = left_comp['volume']

        # If >2 major components, assign extras to nearest kidney
        if len(major) > 2:
            report['confidence'] = 'medium'
            report['notes'] = f'{len(major)} major components found; extras merged by proximity'
            for extra in major_sorted[1:-1]:
                dist_to_right = np.linalg.norm(
                    extra['centroid_world'] - right_comp['centroid_world']
                )
                dist_to_left = np.linalg.norm(
                    extra['centroid_world'] - left_comp['centroid_world']
                )
                target_label = (
                    UNIFIED_RIGHT_KIDNEY if dist_to_right < dist_to_left
                    else UNIFIED_LEFT_KIDNEY
                )
                resolved_mask[labeled_array == extra['id']] = target_label

        # Merge minor components into nearest major kidney
        for m in minor:
            dist_to_right = np.linalg.norm(
                m['centroid_world'] - right_comp['centroid_world']
            )
            dist_to_left = np.linalg.norm(
                m['centroid_world'] - left_comp['centroid_world']
            )
            target_label = (
                UNIFIED_RIGHT_KIDNEY if dist_to_right < dist_to_left
                else UNIFIED_LEFT_KIDNEY
            )
            resolved_mask[labeled_array == m['id']] = target_label

    return resolved_mask, report


def process_kits_dataset(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
) -> pd.DataFrame:
    """Process all KiTS23 cases to resolve kidney laterality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    case_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(case_dirs)} case directories in {input_dir}")

    for case_dir in case_dirs:
        # Look for the unified label (from unify_labels step) or original
        mask_path = case_dir / 'unified_label.nii.gz'
        if not mask_path.exists():
            mask_path = case_dir / 'label.nii.gz'
        if not mask_path.exists():
            logger.warning(f"No label file found for {case_dir.name}, skipping")
            continue

        nii = nib.load(str(mask_path))
        data = np.asarray(nii.dataobj).astype(np.int16)
        affine = nii.affine

        # Only process if kidney label (1) is present
        unique_labels = np.unique(data)
        if KITS_KIDNEY_LABEL not in unique_labels:
            logger.info(f"{case_dir.name}: No kidney label (1) found, skipping laterality")
            continue

        resolved_data, report = resolve_laterality(data, affine)
        report['case_id'] = case_dir.name

        # Save resolved mask
        out_case_dir = output_dir / case_dir.name
        out_case_dir.mkdir(parents=True, exist_ok=True)
        out_nii = nib.Nifti1Image(resolved_data, affine, nii.header)
        out_nii.set_data_dtype(np.uint8)
        nib.save(out_nii, str(out_case_dir / 'unified_label.nii.gz'))

        # Copy image if not already there
        img_src = case_dir / 'image.nii.gz'
        img_dst = out_case_dir / 'image.nii.gz'
        if img_src.exists() and not img_dst.exists():
            import shutil
            shutil.copy2(img_src, img_dst)

        reports.append(report)
        logger.info(
            f"Resolved {case_dir.name}: {report['num_components_raw']} components, "
            f"confidence={report['confidence']}"
        )

    df = pd.DataFrame(reports)
    if not df.empty:
        df.to_csv(report_path, index=False)
        logger.info(f"Laterality report saved to {report_path}")

        # Summary statistics
        conf_counts = df['confidence'].value_counts()
        logger.info(f"Confidence distribution:\n{conf_counts}")
        n_low = conf_counts.get('low', 0) + conf_counts.get('error', 0)
        if n_low > 0:
            logger.warning(
                f"{n_low} cases flagged for manual review "
                f"(confidence='low' or 'error')"
            )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve KiTS23 kidney laterality using 3D connected "
                    "component analysis and world-coordinate centroid position."
    )
    parser.add_argument(
        '--input-dir', type=Path, required=True,
        help="Input directory containing KiTS23 cases (with unified_label.nii.gz)"
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help="Output directory for resolved masks"
    )
    parser.add_argument(
        '--report-path', type=Path, required=True,
        help="Path for the laterality resolution report CSV"
    )
    parser.add_argument(
        '--min-component-voxels', type=int, default=MIN_KIDNEY_COMPONENT_VOXELS,
        help=f"Minimum voxels for a 'major' component (default: {MIN_KIDNEY_COMPONENT_VOXELS})"
    )

    args = parser.parse_args()
    process_kits_dataset(args.input_dir, args.output_dir, args.report_path)


if __name__ == "__main__":
    main()
