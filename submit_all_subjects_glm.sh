#!/bin/bash
#SBATCH --job-name=run_glm           # Name of the job
#SBATCH --time=16:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Adjust CPU allocation if needed
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-7

CONFIG_FILE=$1

cd ~/island_semester_project

# Load modules or activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate miplab_env


# List of subjects
SUBS=(sub-P10 sub-P12 sub-P13 sub-P14 sub-P15 sub-P16 sub-P17 sub-P18)



# Use SLURM_ARRAY_TASK_ID to select subject
SUBJECT=${SUBS[$SLURM_ARRAY_TASK_ID]}


# Run Python script
python src/run_subject_level_glm.py --sub "$SUBJECT" --config "$CONFIG_FILE"