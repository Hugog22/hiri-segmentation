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
#SBATCH --partition=long

mkdir -p logs checkpoints

if command -v module &> /dev/null; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
fi

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/hiri-segmentation/venv/bin/activate" ]; then
    source "$HOME/hiri-segmentation/venv/bin/activate"
fi

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

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi

echo "Starting MONAI training: Model=$MODEL | Fold=$FOLD | Config=$CONFIG | GPUs=$NUM_GPUS"

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m src.training.monai_trainer \
    --model $MODEL \
    --config $CONFIG \
    --fold $FOLD \
    --output-dir checkpoints/$MODEL/fold_$FOLD \
    $RESUME
