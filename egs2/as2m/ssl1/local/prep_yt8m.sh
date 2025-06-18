#!/bin/bash
#SBATCH --job-name=yt8m_data_prep
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.yt8m


. ./db.sh
. ./path.sh

set -euo pipefail
DATA_PREP_ROOT=$1
mkdir -p logs
echo $(which python)
python local/data_prep_yt8m.py ${YT8M} ${DATA_PREP_ROOT}
