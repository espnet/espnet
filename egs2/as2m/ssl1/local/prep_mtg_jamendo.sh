#!/bin/bash
#SBATCH --job-name=mtg_jamendo
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=50
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.mtg_jamendo


. ./db.sh
. ./path.sh

set -euo pipefail
# DATA_PREP_ROOT=$1
mkdir -p logs
echo $(which python)
python local/data_prep_mtg_jamendo.py # ${MTG_JAMENDO} ${DATA_PREP_ROOT}
