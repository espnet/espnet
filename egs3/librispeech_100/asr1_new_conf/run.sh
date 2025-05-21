#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=logs/train_global_%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH -p cpu
#SBATCH --account bbjs-delta-cpu

export N_GPU=1
export CMD_BACKEND=slurm

# We can also create submit.sh for the actual command.
./run \
    --gpu 1 \
    --mem 16g \
    --time 24:00:00 \
    --cpus-per-task 4 \
    srun bash -c '
        source path.sh
        python train.py \
            --config train.yaml \
            --collect_stats
    '

# ./run \
#     --cpus-per-task 1 \
#     --time 24:00:00 \
#     --mem 16g \
#     srun bash -c '
#         source path.sh
#         python inference.py \
#             --config inference.yaml
