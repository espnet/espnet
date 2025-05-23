#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=logs/train_global_%j.log
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH --account bbjs-delta-gpu
#SBATCH --gres=gpu:2


source path.sh

# We can also create submit.sh for the actual command.
# python run.py \
#     --train_config conf/train_ctc_gpu2.yaml \
#     --eval_config inference.yaml \
#     # --collect_stats
#     # --train_tokenizer \
srun python train.py \
    --config train_gpu2.yaml
    # --eval_config inference.yaml \
    # --collect_stats
    # --train_tokenizer \