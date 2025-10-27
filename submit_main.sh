#!/bin/bash
#SBATCH --job-name=run_glm
#SBATCH --time=10:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main.out
#SBATCH --error=logs/main.err


cd ~/island_semester_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate miplab_env

# Collect all YAML configs
CONFIG_FILES=(configs/*.yaml)

# Define all subjects
SUBS=(
    sub-P10 sub-P12 sub-P13 sub-P14 sub-P15 sub-P16 sub-P17 sub-P18 sub-P19
    sub-P20 sub-P21 sub-P22 sub-P23 sub-P25 sub-P27 sub-P30 sub-P31
    sub-P34 sub-P35 sub-P36 sub-P37 sub-P41 sub-P42 sub-P43 sub-P44
    sub-P45 sub-P47 sub-P48 sub-P49 sub-P50 sub-P51
)

echo "-----------------------------------------"
echo "Running GLM for all subjects"
echo "Configs: ${CONFIG_FILES[@]}"
echo "Subjects: ${SUBS[@]}"
echo "-----------------------------------------"

# Run all configs for each subject
for SUBJECT in "${SUBS[@]}"; do
    echo "========== Running GLM for $SUBJECT =========="
    python -m run_subject_level_glm --sub "$SUBJECT" --config "${CONFIG_FILES[@]}"
done

# Run group-level GLM analysis
echo "-----------------------------------------"
echo "Running group-level GLM analysis"
echo "-----------------------------------------"
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    # don't run if 'order' in config filename
    if [[ "$CONFIG_FILE" == *"order"* ]]; then
        echo "Skipping group-level GLM for config: $CONFIG_FILE (order config)"
        continue
    fi
    echo "----- Group-level GLM for config: $CONFIG_FILE"
    python -m run_group_level_glm--config "$CONFIG_FILE"
done
# Once all configs are run, run RSA
echo "-----------------------------------------"
echo "Running RSA analysis"
echo "-----------------------------------------"
python -m src.run_rsa