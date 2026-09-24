#!/bin/bash
# ==============================================================================
# HI&RI: Master Pipeline Orchestrator for UPV HPC Cluster
# Author: Hugo (Especialidad IA, UPV)
# ==============================================================================

set -e

echo "========================================================================"
echo "🏥 HI&RI 3D LIVER & KIDNEY SEGMENTATION PIPELINE"
echo "========================================================================"

# Step 0: Set Environment Variables for nnU-Net v2
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"

echo "Environment:"
echo "  nnUNet_raw:          $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results:      $nnUNet_results"
echo ""

# Step 1: Diagnostic Check
echo "--> [1/5] Running Environment Diagnostics..."
python scripts/check_cluster_env.py

# Step 2: Preprocessing
echo "--> [2/5] Running Standardized Preprocessing..."
python -m src.preprocessing.run_preprocessing --config configs/base.yaml

if [ ! -f "data/processed/manifest.csv" ]; then
    echo ""
    echo "========================================================================"
    echo "⚠️ AVISO: No se encontraron datasets descargados en data/raw/."
    echo "========================================================================"
    echo "Para descargar TotalSegmentator v2 automáticamente en data/raw/:"
    echo "  python scripts/download_datasets.py --download-public"
    echo ""
    echo "O si ya tienes datos descargados en otra carpeta del cluster, enlázalos:"
    echo "  ln -s /ruta/a/tus/datos/TotalSegmentator data/raw/TotalSegmentator"
    echo "  ln -s /ruta/a/tus/datos/KiTS23 data/raw/KiTS23"
    echo ""
    echo "Una vez tengas los datos en data/raw/, vuelve a ejecutar:"
    echo "  ./scripts/run_cluster_pipeline.sh"
    echo "========================================================================"
    exit 0
fi

# Step 3: Splits and Anti-Leakage Verification
echo "--> [3/5] Generating 5-Fold Stratified Splits & Verifying Anti-Leakage..."
python -m src.splits.create_splits \
    --manifest data/processed/manifest.csv \
    --config configs/base.yaml \
    --output data/splits.json

python -m src.splits.verify_no_leakage \
    --splits-json data/splits.json \
    --manifest data/processed/manifest.csv

# Step 4: Prepare nnU-Net v2 Dataset
echo "--> [4/5] Formatting Dataset for nnU-Net v2 (Dataset100_HIRI)..."
python -m src.preprocessing.prepare_nnunet_dataset \
    --input-dir data/processed \
    --output-dir nnUNet_raw \
    --splits-json data/splits.json \
    --dataset-id 100

echo ""
echo "========================================================================"
echo "✅ PREPROCESSING & SETUP COMPLETE!"
echo "========================================================================"
echo "To launch training jobs on the SLURM cluster:"
echo ""
echo "  A) Train MONAI Swin UNETR (5 folds in parallel):"
echo "     sbatch scripts/slurm/train_folds_parallel.sbatch"
echo ""
echo "  B) Plan and Train nnU-Net v2 ResEnc L:"
echo "     python -m src.training.run_nnunet plan --dataset-id 100 --planner nnUNetPlannerResEncL"
echo "     sbatch scripts/slurm/train_nnunet.sbatch"
echo ""
echo "  C) Run Full Benchmark & Statistical Analysis (after training):"
echo "     python -m src.evaluation.run_benchmark --gt-dir data/processed --models-json configs/models_eval.json --output-dir results/benchmark"
echo "========================================================================"
