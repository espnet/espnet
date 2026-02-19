#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=asr_female
#SBATCH --output=slurm_female_%j.log
#SBATCH --error=slurm_female_%j.log

module load ffmpeg/6.1.1
module load cuda/12.4.1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate espnet

cd ~/bootcamp/espnet/egs2/commonvoice_gender/asr1
. ./path.sh

# Female pipeline: collect stats (10), train (11), decode (12), score (13)
./run_female.sh --stage 10 --stop-stage 13
