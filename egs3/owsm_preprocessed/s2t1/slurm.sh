#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=logs/train_owsm_%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -p <>
#SBATCH --account <>
#SBATCH -J owsm_gpu1
#SBATCH --gres=gpu:1


source path.sh

# We can also create submit.sh for the actual command.
python train.py --config train.yaml