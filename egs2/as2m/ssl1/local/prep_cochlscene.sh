#!/bin/bash
#SBATCH --job-name=cochlscene_data_prep
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=70
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.cochlscene


. ./db.sh
. ./path.sh

set -euo pipefail
DATA_PREP_ROOT=$1
PARALLELISM=64
mkdir -p logs
echo $(which python)
python local/data_prep_cochlscene.py ${COCHLSCENE} ${DATA_PREP_ROOT} ${PARALLELISM}
