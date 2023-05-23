#!/usr/bin/env bash

# Calculate accuracy of InterCTC-based Aphasia detection

set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
decode_dir=
layer_idx=

if [ -z "${decode_dir}" ] || [ -z "${layer_idx}" ]; then
  log "Specify --decode_dir and --layer_idx"
  exit 2
fi

log "$0 $*"
. utils/parse_options.sh

help_message=$(
  cat <<EOF
Usage: $0

Options:
    --decode_dir (string): Path to decoded results.
      For example, --decode_dir=exp/asr_train_asr_conformer_raw_en_char_sp/decode_asr_model_valid.acc.ave/test"
    --layer_idx (int): Index of the layer to be used for InterCTC-based Aphasia detection.
EOF
)

if [ -z "${decode_dir}" ]; then
  log "${help_message}"
  exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  cat "${decode_dir}"/logdir/output.*/1best_recog/"encoder_interctc_layer${layer_idx}.txt" >"${decode_dir}"/interctc.txt
  python local/score_aphasia_detection.py "${decode_dir}"/interctc.txt
fi
