#!/bin/bash
#SBATCH --job-name=run_rsa           # Name of the job
#SBATCH --time=16:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Adjust CPU allocation if needed
#SBATCH --output=logs/rsa_job.out    # Output log file
#SBATCH --error=logs/rsa_job.err     # Error log file

cd ~/island_semester_project

# Load modules or activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate miplab_env

# Run your Python script
python src/run_correlation.py 
