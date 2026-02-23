#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH--time=24:00:00 

source path.sh

python run.py \
    --stages infer measure \
    --train_config conf/tuning/train_e_branchformer.yaml \
    --infer_config conf/inference.yaml \
    --measure_config conf/metric.yaml 
