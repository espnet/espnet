#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=male_on_female
#SBATCH --output=slurm_male_on_female_%j.log
#SBATCH --error=slurm_male_on_female_%j.log

module load ffmpeg/6.1.1
module load cuda/12.4.1

# Get the directory where this script is located
cd "$(dirname "$0")"
. ./path.sh

./decode_cross_gender.sh male_on_female
