#!/bin/bash
#SBATCH --job-name=run_glm           # Name of the job
#SBATCH --time=00:20:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Adjust CPU allocation if needed
#SBATCH --output=logs/glm_%A_%a.out
#SBATCH --error=logs/glm_%A_%a.err

# In order to run all subjects submit -# sbatch --array=0-9 ..., sbatch --array=10-19 ..., sbatch --array=20-29 ..., sbatch --array=30-31 ...

CONFIG_FILE=$1

cd ~/island_semester_project

# Load modules or activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate miplab_env


# List of subjects
SUBS=(sub-P10 sub-P12 sub-P13 sub-P14 sub-P15 sub-P16 sub-P17 sub-P18 sub-P19 sub-P20 sub-P21 sub-P22 sub-P23 sub-P25 sub-P26 sub-P27 sub-P30 sub-P31 sub-P34 sub-P35 sub-P36 sub-P37 sub-P41 sub-P42 sub-P43 sub-P44 sub-P45 sub-P47 sub-P48 sub-P49 sub-P50 sub-P51)

if [ $SLURM_ARRAY_TASK_ID -ge ${#SUBS[@]} ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds number of subjects (${#SUBS[@]})"
    exit 1
fi

# Use SLURM_ARRAY_TASK_ID to select subject
SUBJECT=${SUBS[$SLURM_ARRAY_TASK_ID]}


# Run Python script
python src/run_subject_level_glm.py --sub "$SUBJECT" --config "$CONFIG_FILE"