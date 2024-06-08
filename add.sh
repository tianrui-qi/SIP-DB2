#!/bin/bash
#SBATCH --job-name=selector
#SBATCH --output=/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/out/%A_%a.out
#SBATCH --error=/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/err/%A_%a.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --array=0-15

# Activate your virtual environment if needed
source /home/s.tianrui.qi/miniconda3/etc/profile.d/conda.sh
conda activate SIP-DB2

# Run the Python script with the task ID
srun python add.py $SLURM_ARRAY_TASK_ID
