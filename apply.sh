#!/bin/bash
#SBATCH --job-name=selector
#SBATCH --output=/home/s.tianrui.qi/SIP-DB2/data/out/%A_%a.out
#SBATCH --error=/home/s.tianrui.qi/SIP-DB2/data/err/%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --array=1-32

# Activate your virtual environment if needed
source /home/s.tianrui.qi/miniconda3/etc/profile.d/conda.sh
conda activate SIP-DB2

# Run the Python script with the task ID
srun python apply.py $SLURM_ARRAY_TASK_ID
