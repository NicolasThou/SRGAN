#!/bin/bash

#
# Slurm arguments
#
#SBATCH --ntasks=1
#SBATCH --job-name "srgan"
#SBATCH --mem-per-cpu=12000      # Memory to allocate in MB per allocated CPU core
#SBATCH --output=output.txt      # output 
#SBATCH --gres=gpu:1             # Number of GPU's
#SBATCH --time="7-00:00:00"      # Max execution time

module load python
python train.py
