#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail


log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
  }
SECONDS=0

stage=1
stop_stage=100

trans_type=

datasets_root=${LJSPEECH}

log "$0 $*"
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

train_set="${trans_type}_train_nodev"
dev_set="${trans_type}_dev"
eval_sets="${trans_type}_eval"

cwd=$(pwd)
if [ ! -e "${datasets_root}/LJSpeech-1.1" ]; then
  log "Stage 0: Data download."
  local/data_download.sh ${datasets_root}
else
  echo "LJSpeech already exists. Skipped."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: Data preparation."
  local/data_prep.sh ${datasets_root}/LJSpeech-1.1 data/${trans_type}_train ${trans_type}
  utils/validate_data_dir.sh --no-feats data/${trans_type}_train
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 2: Prepare training set, dev set and eval set."
  mkdir -p data/${train_set} data/${dev_set} data/${eval_set}
  
  utils/subset_data_dir.sh --last data/${trans_type}_train 500 data/${trans_type}_deveval
  utils/subset_data_dir.sh --last data/${trans_type}_deveval 250 data/${eval_sets}
  utils/subset_data_dir.sh --first data/${trans_type}_deveval 250 data/${dev_set}
  n=$(( $(wc -l < data/${trans_type}_train/wav.scp) - 500 ))
  utils/subset_data_dir.sh --first data/${trans_type}_train ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
  


  
  








