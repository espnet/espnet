#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=logs/train_global_%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -p gpuA40x4
#SBATCH --account bbjs-delta-gpu
#SBATCH --gres=gpu:1

. path.sh

export N_GPU=1
export WANDB_API_KEY=<>

wandb login

srun python ./train.py
srun python ./inference.py \
    --config_path config.yaml \
    --inference_config inference.py