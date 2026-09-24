"""
Tests for orientation verification and correction.
"""
import pytest
import nibabel as nib
import numpy as np
from pathlib import Path

def get_orientation(affine: np.ndarray) -> str:
    return "".join(nib.aff2axcodes(affine))

def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    orig_ornt = nib.io_orientation(img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    new_data = nib.orientations.apply_orientation(img.get_fdata(), transform)
    new_affine = img.affine.dot(nib.orientations.inv_ornt_aff(transform, img.shape))
    return nib.Nifti1Image(new_data, new_affine)

def check_consistency(img: nib.Nifti1Image, mask: nib.Nifti1Image) -> bool:
    if img.shape != mask.shape:
        return False
    if not np.allclose(img.affine, mask.affine):
        return False
    return True

def test_ras_image_passes():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10,10,10)), affine)
    assert get_orientation(img.affine) == 'RAS'

def test_lps_image_detected():
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10,10,10)), affine)
    assert get_orientation(img.affine) == 'LPS'

def test_reorient_lps_to_ras():
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10,10,10)), affine)
    reoriented = reorient_to_ras(img)
    assert get_orientation(reoriented.affine) == 'RAS'

def test_reorient_preserves_data():
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    data = np.arange(8).reshape(2,2,2)
    img = nib.Nifti1Image(data, affine)
    reoriented = reorient_to_ras(img)
    assert np.sum(reoriented.get_fdata()) == np.sum(data)
    assert reoriented.shape == img.shape

def test_image_mask_consistency_pass():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10,10,10)), affine)
    mask = nib.Nifti1Image(np.zeros((10,10,10)), affine)
    assert check_consistency(img, mask) is True

def test_image_mask_consistency_fail_shape():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(np.zeros((10,10,10)), affine)
    mask = nib.Nifti1Image(np.zeros((10,10,11)), affine)
    assert check_consistency(img, mask) is False

def test_image_mask_consistency_fail_affine():
    img = nib.Nifti1Image(np.zeros((10,10,10)), np.diag([1.0, 1.0, 1.0, 1.0]))
    mask = nib.Nifti1Image(np.zeros((10,10,10)), np.diag([2.0, 1.0, 1.0, 1.0]))
    assert check_consistency(img, mask) is False
