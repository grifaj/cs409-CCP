#!/bin/bash
#
#SBATCH --job-name=sealScriptProject # Job name for tracking
#SBATCH --partition=gecko     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=job_logs/train_job_%j.out # Standard out from your job
#SBATCH --error=job_logs/train_job_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA
conda activate seals 

srun /dcs/20/u2002183/.conda/envs/seals/bin/python /dcs/20/u2002183/cs409-CCP-1/data/train.py 
