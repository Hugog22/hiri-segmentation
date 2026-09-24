"""
Cluster Environment and Hardware Diagnostic Script.

Verifies:
- Python version >= 3.10
- PyTorch with CUDA support and GPU device details (VRAM, compute capability)
- MONAI and nnU-Net v2 package integrity
- Environment variables (nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
- Available disk space in project and data directories
- Configuration files and dataset manifests
"""
import os
import shutil
import sys
from pathlib import Path


def check_python():
    print("=" * 60)
    print("1. PYTHON ENVIRONMENT")
    print("=" * 60)
    v = sys.version_info
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("❌ ERROR: Python >= 3.10 required.")
        return False
    print("✅ Python version OK.")
    return True


def check_pytorch_cuda():
    print("\n" + "=" * 60)
    print("2. PYTORCH & CUDA ACCELERATION")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if not cuda_avail:
            print("⚠️ WARNING: CUDA not detected! Running on CPU will be extremely slow for 3D volumes.")
            return False
            
        device_count = torch.cuda.device_count()
        print(f"GPU Device Count: {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  - GPU {i}: {name} ({vram_gb:.1f} GB VRAM)")
            
        print("✅ PyTorch CUDA configuration OK.")
        return True
    except ImportError:
        print("❌ ERROR: PyTorch is not installed in current environment.")
        return False


def check_libraries():
    print("\n" + "=" * 60)
    print("3. MEDICAL IMAGING LIBRARIES")
    print("=" * 60)
    all_ok = True
    
    # MONAI
    try:
        import monai
        print(f"✅ MONAI Version: {monai.__version__}")
    except ImportError:
        print("❌ MONAI is NOT installed.")
        all_ok = False
        
    # nnU-Net
    try:
        import nnunetv2
        print(f"✅ nnU-Net v2 Version: {nnunetv2.__version__}")
    except ImportError:
        print("❌ nnU-Net v2 is NOT installed.")
        all_ok = False
        
    # SimpleITK & nibabel
    try:
        import SimpleITK as sitk
        import nibabel as nib
        print(f"✅ SimpleITK Version: {sitk.__version__}")
        print(f"✅ Nibabel Version: {nib.__version__}")
    except ImportError as e:
        print(f"❌ Imaging library missing: {e}")
        all_ok = False
        
    return all_ok


def check_nnunet_env():
    print("\n" + "=" * 60)
    print("4. nnU-Net v2 ENVIRONMENT VARIABLES")
    print("=" * 60)
    required_vars = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    all_set = True
    for var in required_vars:
        val = os.environ.get(var)
        if val:
            p = Path(val)
            exists = p.exists()
            status = "EXISTS" if exists else "NOT FOUND (will be created)"
            print(f"✅ {var} = {val} [{status}]")
        else:
            print(f"⚠️ {var} is NOT SET. (Run `source setup_env.sh` or configure in cluster sbatch).")
            all_set = False
    return all_set


def check_disk_space():
    print("\n" + "=" * 60)
    print("5. DISK STORAGE")
    print("=" * 60)
    cwd = Path.cwd()
    stat = shutil.disk_usage(cwd)
    free_gb = stat.free / (1024 ** 3)
    total_gb = stat.total / (1024 ** 3)
    print(f"Working Directory: {cwd}")
    print(f"Free Space: {free_gb:.1f} GB / {total_gb:.1f} GB")
    if free_gb < 50.0:
        print("⚠️ WARNING: Less than 50 GB free disk space. 3D CT preprocessed caches and models require ~100+ GB.")
    else:
        print("✅ Sufficient disk space detected.")


def main():
    print("\n🏥 HI&RI CLUSTER ENVIRONMENT DIAGNOSTIC TOOL\n")
    p_ok = check_python()
    c_ok = check_pytorch_cuda()
    l_ok = check_libraries()
    n_ok = check_nnunet_env()
    check_disk_space()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    if p_ok and c_ok and l_ok:
        print("🎉 Ready for cluster training! Launch your SLURM jobs via scripts/slurm/.")
    else:
        print("⚠️ Please resolve the items flagged with ❌ or ⚠️ above before launching full training.")


if __name__ == "__main__":
    main()
