#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=asr_decode
#SBATCH --output=slurm_decode_%j.log
#SBATCH --error=slurm_decode_%j.log

module load ffmpeg/6.1.1
module load cuda/12.4.1

# Get the directory where this script is located
cd "$(dirname "$0")"
. ./path.sh

./run.sh --stage 12 --stop-stage 13
