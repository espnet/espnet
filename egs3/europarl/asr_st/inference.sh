#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=logs/train_global_%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -p cpu
#SBATCH --account bbjs-delta-cpu

. path.sh

srun python ./inference.py --config inference.yaml
