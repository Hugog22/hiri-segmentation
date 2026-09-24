"""
Automated download and directory preparation helper for public datasets.

Handles:
- TotalSegmentator v2 (Zenodo)
- KiTS23 (GitHub / clone & download)
- LiTS (CodaLab verification)
- AMOS22 (Zenodo / Grand-Challenge)
- BTCV (Synapse guidance)
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIRECTORIES = [
    "data/raw/TotalSegmentator",
    "data/raw/LiTS",
    "data/raw/KiTS23",
    "data/raw/AMOS22",
    "data/raw/BTCV",
    "data/processed",
    "nnUNet_raw",
    "nnUNet_preprocessed",
    "nnUNet_results"
]


def setup_directories(base_dir: Path) -> None:
    """Create all standard directory hierarchies."""
    for rel_dir in DATASET_DIRECTORIES:
        p = base_dir / rel_dir
        p.mkdir(parents=True, exist_ok=True)
    logger.info("Created directory structure for raw, processed, and nnU-Net folders.")


def download_totalsegmentator(dest_dir: Path) -> None:
    """Download TotalSegmentator v2 dataset from Zenodo."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zenodo_url = "https://zenodo.org/records/6802614/files/Totalsegmentator_dataset_v201.zip"
    logger.info("Downloading TotalSegmentator v2 from Zenodo...")
    zip_path = dest_dir / "Totalsegmentator_dataset.zip"
    if zip_path.exists():
        logger.info("TotalSegmentator zip archive already exists. Skipping download.")
        return
        
    cmd = ["curl", "-L", "-o", str(zip_path), zenodo_url]
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Downloaded TotalSegmentator archive to {zip_path}")
        logger.info("Extracting...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(dest_dir)], check=True)
        logger.info("Extraction complete.")
    except Exception as e:
        logger.error(f"Download or unzip failed: {e}")


def download_kits23(dest_dir: Path) -> None:
    """Setup KiTS23 repository and trigger its download script."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    kits_repo = dest_dir / "kits23_repo"
    if not kits_repo.exists():
        logger.info("Cloning official KiTS23 GitHub repository...")
        subprocess.run(["git", "clone", "https://github.com/neheller/kits23.git", str(kits_repo)], check=True)
    logger.info(f"KiTS23 repo ready at {kits_repo}. Follow official instructions to download CT volumes.")


def print_manual_dataset_instructions() -> None:
    """Print instructions for datasets requiring portal authentication."""
    print("=" * 70)
    print("INSTRUCCIONES DE DESCARGA PARA DATASETS CON AUTENTICACIÓN")
    print("=" * 70)
    print("""
1. LiTS (Liver Tumor Segmentation):
   - Portal: https://competitions.codalab.org/competitions/17094
   - Inicia sesión y descarga 'Training Batch 1' y 'Training Batch 2' en data/raw/LiTS/

2. AMOS22:
   - Portal: https://amos22.grand-challenge.org/ o Zenodo: https://zenodo.org/records/7155725
   - Descarga 'amos22.zip' y descomprime en data/raw/AMOS22/

3. BTCV (Synapse Multi-Atlas):
   - Portal: https://www.synapse.org/#!Synapse:syn3193805
   - Acepta el Data Use Agreement de Vanderbilt y descarga 'RawData.zip' en data/raw/BTCV/
""")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset setup helper.")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base repository path")
    parser.add_argument("--download-public", action="store_true", help="Download open-access datasets (TotalSegmentator, KiTS23 repo)")
    args = parser.parse_args()

    setup_directories(args.base_dir)
    print_manual_dataset_instructions()

    if args.download_public:
        download_totalsegmentator(args.base_dir / "data/raw/TotalSegmentator")
        download_kits23(args.base_dir / "data/raw/KiTS23")


if __name__ == "__main__":
    main()
