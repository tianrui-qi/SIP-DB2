#!/bin/bash
#SBATCH --job-name=temp_embd2repre
#SBATCH --output=data/out/%A_%a.out
#SBATCH --error=data/err/%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --array=0-22

# Activate your virtual environment if needed
source /home/s.tianrui.qi/miniconda3/etc/profile.d/conda.sh
conda activate SIP-DB2

# Run the Python script with the task ID
srun python temp_embd2repre.py $SLURM_ARRAY_TASK_ID
