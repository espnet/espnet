#!/bin/bash
#SBATCH --nodes=2
#SBATCH --output=logs/debug_%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00 
#SBATCH -p general
#SBATCH --gres=gpu:1

. path.sh

srun python egs3/train.py
