#!/bin/bash
#SBATCH --job-name=eval_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=long

mkdir -p logs results/benchmark

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/hiri-segmentation/venv/bin/activate" ]; then
    source "$HOME/hiri-segmentation/venv/bin/activate"
fi

export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Iniciando benchmark y evaluacion clinica..."
python -m src.evaluation.run_benchmark \
    --gt-dir nnUNet_raw/Dataset100_HIRI/labelsTs \
    --models-json configs/models_eval.json \
    --output-dir results/benchmark \
    --baseline nnUNet_ResEncL_Ensemble

echo "Benchmark finalizado con exito."
