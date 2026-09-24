#!/bin/bash
#SBATCH --job-name=train_monai
#SBATCH --output=logs/monai_%j.out
#SBATCH --error=logs/monai_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu

module load cuda/12.1

# Activate environment (adjust path if needed)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate hiri-env
# OR if using apptainer, prepend apptainer exec to the command

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

MODEL="swin_unetr_base"
CONFIG="configs/base.yaml"
FOLD=0
RESUME=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        --fold) FOLD="$2"; shift ;;
        --resume) RESUME="--resume $2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting MONAI training: Model=$MODEL | Fold=$FOLD | Config=$CONFIG"

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m src.training.monai_trainer \
    --model $MODEL \
    --config $CONFIG \
    --fold $FOLD \
    --output-dir checkpoints/$MODEL/fold_$FOLD \
    $RESUME
