"""
Tests for resampling.
"""
import pytest
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json

def resample_image(image: sitk.Image, new_spacing: tuple[float, float, float], is_label: bool = False) -> sitk.Image:
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    
    new_size = [
        int(round(orig_size[0] * (orig_spacing[0] / new_spacing[0]))),
        int(round(orig_size[1] * (orig_spacing[1] / new_spacing[1]))),
        int(round(orig_size[2] * (orig_spacing[2] / new_spacing[2])))
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        
    return resampler.Execute(image)

def create_sitk_image(shape=(10,10,10), spacing=(1.0, 1.0, 2.0), is_label=False):
    data = np.random.randint(0, 4, size=shape).astype(np.uint8) if is_label else np.random.rand(*shape).astype(np.float32)
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(spacing)
    return img

def test_resample_changes_spacing():
    img = create_sitk_image(spacing=(1.0, 1.0, 2.0))
    resampled = resample_image(img, (1.0, 1.0, 1.5))
    assert np.allclose(resampled.GetSpacing(), (1.0, 1.0, 1.5))

def test_resample_mask_nearest():
    img = create_sitk_image(is_label=True)
    resampled = resample_image(img, (1.5, 1.5, 1.5), is_label=True)
    arr = sitk.GetArrayFromImage(resampled)
    assert set(np.unique(arr)).issubset({0, 1, 2, 3})

def test_resample_preserves_label_set():
    img = create_sitk_image(is_label=True)
    orig_labels = set(np.unique(sitk.GetArrayFromImage(img)))
    resampled = resample_image(img, (1.5, 1.5, 1.5), is_label=True)
    new_labels = set(np.unique(sitk.GetArrayFromImage(resampled)))
    assert new_labels.issubset(orig_labels)

def test_resample_image_interpolation():
    img = create_sitk_image(is_label=False)
    orig_vals = set(np.unique(sitk.GetArrayFromImage(img)))
    resampled = resample_image(img, (1.5, 1.5, 1.5), is_label=False)
    new_vals = set(np.unique(sitk.GetArrayFromImage(resampled)))
    assert not new_vals.issubset(orig_vals)

def test_resample_identity():
    img = create_sitk_image(spacing=(1.0, 1.0, 2.0))
    resampled = resample_image(img, (1.0, 1.0, 2.0), is_label=False)
    arr1 = sitk.GetArrayFromImage(img)
    arr2 = sitk.GetArrayFromImage(resampled)
    assert np.allclose(arr1, arr2)

def test_original_metadata_saved(tmp_path: Path):
    img = create_sitk_image(spacing=(1.0, 1.0, 2.0))
    meta = {"original_spacing": img.GetSpacing()}
    out_path = tmp_path / "meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f)
    
    with open(out_path, "r") as f:
        loaded = json.load(f)
    assert tuple(loaded["original_spacing"]) == (1.0, 1.0, 2.0)
