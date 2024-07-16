#!/bin/bash
#SBATCH --job-name=apply
#SBATCH --output=data/out/%A_%a.out
#SBATCH --error=data/err/%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --array=0-68

# Activate your virtual environment if needed
source /home/s.tianrui.qi/miniconda3/etc/profile.d/conda.sh
conda activate SIP-DB2

# Run the Python script with the task ID
srun python apply.py $SLURM_ARRAY_TASK_ID
