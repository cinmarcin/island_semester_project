#!/bin/bash
#SBATCH --job-name=run_glm
#SBATCH --time=03:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Capture all arguments as config files (can be one or more)
CONFIG_FILES=("$@")
LOG_DIR=logs
mkdir -p "$LOG_DIR"

# Create a descriptive name for logs (join config base names)
CONFIG_NAMES=""
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
    CONFIG_NAMES+="${CONFIG_BASENAME}_"
done
CONFIG_NAMES=${CONFIG_NAMES%_}  # remove trailing underscore

LOG_OUT=${LOG_DIR}/glm_${CONFIG_NAMES}_job${SLURM_ARRAY_JOB_ID}_array${SLURM_ARRAY_TASK_ID}.out
LOG_ERR=${LOG_DIR}/glm_${CONFIG_NAMES}_job${SLURM_ARRAY_JOB_ID}_array${SLURM_ARRAY_TASK_ID}.err

# Redirect all output to custom logs
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

# Activate environment
cd ~/island_semester_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate miplab_env

# Define subjects
SUBS=(sub-P10 sub-P12 sub-P13 sub-P14 sub-P15 sub-P16 sub-P17 sub-P18 sub-P19 sub-P20 sub-P21 sub-P22 sub-P23 sub-P25 sub-P27 sub-P30 sub-P31 sub-P34 sub-P35 sub-P36 sub-P37 sub-P41 sub-P42 sub-P43 sub-P44 sub-P45 sub-P47 sub-P48 sub-P49 sub-P50 sub-P51)

# Split subjects across array jobs
SUBS_PER_TASK=16
START=$((SLURM_ARRAY_TASK_ID * SUBS_PER_TASK))
END=$((START + SUBS_PER_TASK - 1))

if [ $START -ge ${#SUBS[@]} ]; then
    echo "No subjects for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi
if [ $END -ge ${#SUBS[@]} ]; then
    END=$((${#SUBS[@]} - 1))
fi

echo "Task $SLURM_ARRAY_TASK_ID running subjects ${SUBS[@]:$START:$((END-START+1))}"
echo "Using config files: ${CONFIG_FILES[@]}"

# Run GLM for assigned subjects
for ((i=$START; i<=$END; i++)); do
    SUBJECT=${SUBS[$i]}
    echo "Running GLM for $SUBJECT"
    python src/run_subject_level_glm.py --sub "$SUBJECT" --config "${CONFIG_FILES[@]}"
done
