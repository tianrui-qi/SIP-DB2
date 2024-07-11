#!/bin/bash
#SBATCH --job-name=selector
#SBATCH --output=data/out/%A_%a.out
#SBATCH --error=data/err/%A_%a.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --array=0-86

# Activate your virtual environment if needed
source /home/s.tianrui.qi/miniconda3/etc/profile.d/conda.sh
conda activate SIP-DB2

# Run the Python script with the task ID
srun python add.py $SLURM_ARRAY_TASK_ID
