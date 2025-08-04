#!/usr/bin/env bash

set -e
set -u
set -o pipefail

export HF_HUB_OFFLINE=1

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

# right after you set it:
log "HF_HUB_OFFLINE is set to: ${HF_HUB_OFFLINE}"

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


train_set="train"
dev_set="val"
test_sets="test"

db_root=${GLOBE}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download (HugginFace CLI Download Separately)"
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Data preparation via HuggingFace Datasets (batch)"
  _dp_logdir=logs/data_prep
  mkdir -p "${_dp_logdir}"
  # Submit a single job via train_cmd; logs handled by slurm.pl with logfile mapping
  ${train_cmd} JOB=1:1 "${_dp_logdir}"/data_prep.JOB.log \
    python3 local/data_prep.py \
      --train_set "${train_set}" \
      --dev_set   "${dev_set}"   \
      --test_set  "${test_sets}" \
      --hf_repo   "MushanW/GLOBE_V2" \
      --dest_path "data" \
      --jobs      4
fi

utils/fix_data_dir.sh data/${train_set}
utils/fix_data_dir.sh data/${dev_set}
utils/fix_data_dir.sh data/${test_sets}


unset HF_HUB_OFFLINE

log "HF_HUB_OFFLINE is now: ${HF_HUB_OFFLINE-<unset>}"