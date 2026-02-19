#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=asr_train
#SBATCH --output=slurm_train_%j.log
#SBATCH --error=slurm_train_%j.log

module load ffmpeg/6.1.1
module load cuda/12.4.1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate espnet

cd ~/bootcamp/espnet/egs2/commonvoice_gender/asr1
. ./path.sh

./run.sh --stage 11 --stop-stage 13
