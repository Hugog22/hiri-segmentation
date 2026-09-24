# HI&RI - Hepatic and Renal Image Segmentation for Transplant Volumetry

3D medical image segmentation project for liver and kidney volumetric analysis from abdominal CT scans. This project unifies multiple datasets (TotalSegmentator, LiTS, KiTS23, AMOS22, BTCV) and leverages state-of-the-art frameworks including MONAI and nnU-Net v2.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Download data**:
   Follow instructions in `data/README.md`.

3. **Configure parameters**:
   Edit `configs/base.yaml` to suit your requirements.

4. **Train**:
   Submit SLURM scripts located in `scripts/slurm/`.

## Project Structure
```
.
├── configs/            # Configuration files (YAML)
├── data/               # Raw and processed datasets (ignored by git)
│   └── README.md       # Dataset download instructions
├── scripts/            # Bash and SLURM scripts
├── src/                # Python source code
│   ├── app/            # CLI and entry points
│   ├── evaluation/     # Metrics and statistical tests
│   ├── inference/      # Sliding window and post-processing
│   ├── preprocessing/  # Resampling, normalization
│   ├── splits/         # Cross-validation splits
│   └── training/       # Training loops and models
├── tests/              # Unit tests
├── apptainer.def       # Apptainer container definition
├── pyproject.toml      # Python dependencies and metadata
└── README.md           # This file
```

## Citation
If you use this project, please cite... (Placeholder)

## License
(Placeholder)
