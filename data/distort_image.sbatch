#!/bin/bash
#
#SBATCH --job-name=sealscript-distortims # Job name for tracking
#SBATCH --partition=parrot  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=1      # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem-per-cpu=2000     # RAM in MB needed per core (i.e. 1500MB * 40 threads = 60GB RAM)
#SBATCH --time=0-01:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

conda activate seals
## Execute your program(s) ##
srun python /dcs/20/u2009169/Documents/sealScript/cs409-CCP/data/distort_image_batch.py
## Other examples:
## A Python program
# srun python3 my_program.py
#
## Excutable progra