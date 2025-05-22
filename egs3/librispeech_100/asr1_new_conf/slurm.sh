#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=logs/train_global_%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -p <>
#SBATCH --account <>
#SBATCH --gres=gpu:1


source path.sh

# We can also create submit.sh for the actual command.
python run.py \
    --train_config train.yaml \
    --eval_config inference.yaml \
    --train_tokenizer \
    --collect_stats
