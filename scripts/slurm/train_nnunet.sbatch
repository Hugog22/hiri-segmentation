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
# #SBATCH --partition=gpu  # Ajustar con -p <particion> segun salida de sinfo

mkdir -p logs

if command -v module &> /dev/null; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
fi

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/hiri-segmentation/venv/bin/activate" ]; then
    source "$HOME/hiri-segmentation/venv/bin/activate"
fi

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

# Normalize planner class name to plans identifier
if [ "$PLANNER" == "nnUNetPlannerResEncL" ] || [ "$PLANNER" == "nnUNetResEncUNetLPlans" ]; then
    PLANS="nnUNetResEncUNetLPlans"
elif [ "$PLANNER" == "nnUNetPlannerResEncXL" ] || [ "$PLANNER" == "nnUNetResEncUNetXLPlans" ]; then
    PLANS="nnUNetResEncUNetXLPlans"
elif [ "$PLANNER" == "nnUNetPlanner" ] || [ "$PLANNER" == "nnUNetPlans" ]; then
    PLANS="nnUNetPlans"
else
    PLANS="$PLANNER"
fi

PREP_DIR="${nnUNet_preprocessed}/Dataset${DATASET_ID}_HIRI"
if [ ! -f "${PREP_DIR}/${PLANS}.json" ]; then
    DETECTED_PLANS=$(find "$PREP_DIR" -maxdepth 1 -name "*Plans.json" 2>/dev/null | head -n 1 | xargs -r basename -s .json)
    if [ -n "$DETECTED_PLANS" ]; then
        echo "Auto-detected plans identifier: $DETECTED_PLANS"
        PLANS="$DETECTED_PLANS"
    fi
fi

echo "Starting nnU-Net v2 training for dataset $DATASET_ID, config $CONFIG, fold $FOLD, plans $PLANS"

# Run nnUNetv2_train for the specified fold
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -p $PLANS $RESUME
