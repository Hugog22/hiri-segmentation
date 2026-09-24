#!/bin/bash
#SBATCH --job-name=train_folds
#SBATCH --array=0-4
#SBATCH --output=logs/folds_%A_%a.out
#SBATCH --error=logs/folds_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu

MODEL_TYPE="monai"
MODEL_NAME="swin_unetr_base"
PLANNER="nnUNetPlannerResEncL"
CONFIG="configs/base.yaml"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-type) MODEL_TYPE="$2"; shift ;;
        --model-name) MODEL_NAME="$2"; shift ;;
        --planner) PLANNER="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

FOLD=$SLURM_ARRAY_TASK_ID

echo "=== SLURM Array Task: Fold $FOLD | Model Type: $MODEL_TYPE | Model: $MODEL_NAME ==="

if [ "$MODEL_TYPE" == "nnunet" ]; then
    bash scripts/slurm/train_nnunet.sbatch --fold $FOLD --planner $PLANNER
elif [ "$MODEL_TYPE" == "monai" ]; then
    bash scripts/slurm/train_monai.sbatch --fold $FOLD --model $MODEL_NAME --config $CONFIG
else
    echo "Unknown model type: $MODEL_TYPE (choose 'monai' or 'nnunet')"
    exit 1
fi
