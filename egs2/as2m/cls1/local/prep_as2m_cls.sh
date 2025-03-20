#!/bin/bash
#SBATCH --job-name=as2m_cls
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=local/logs/%j.as2m_cls.log


. ./db.sh
. ./path.sh

set -euo pipefail
DATA_PREP_ROOT=$1
mkdir -p local/logs
echo $(which python)

python local/data_prep_as2m.py ${AUDIOSET} ${DATA_PREP_ROOT}