#!/bin/bash
#SBATCH --job-name=wavcaps_data_prep
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=76
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.wavcaps


. ./db.sh
. ./path.sh

PARALLELISM=64
DATA_PREP_ROOT=/work/nvme/bbjs/sbharadwaj/fullas2m/data
mkdir -p logs
echo $(which python)
python local/data_prep_wavcaps.py ${WAVCAPS} ${DATA_PREP_ROOT} ${PARALLELISM}