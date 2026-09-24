#!/bin/bash
#SBATCH --job-name=train_nnunet
#SBATCH --output=logs/nnunet_%j.out
#SBATCH --error=logs/nnunet_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu

module load cuda/12.1

export nnUNet_raw="${nnUNet_raw:-${PWD}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PWD}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PWD}/nnUNet_results}"

DATASET_ID="100"
FOLD=0
CONFIG="3d_fullres"
PLANNER="nnUNetPlannerResEncL"
RESUME=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET_ID="$2"; shift ;;
        --fold) FOLD="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        --planner) PLANNER="$2"; shift ;;
        --resume) RESUME="-c"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting nnU-Net v2 training for dataset $DATASET_ID, config $CONFIG, fold $FOLD, planner $PLANNER"

# Run nnUNetv2_train with 4 GPUs (or 1 depending on cluster partition)
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -p $PLANNER -num_gpus 4 $RESUME
