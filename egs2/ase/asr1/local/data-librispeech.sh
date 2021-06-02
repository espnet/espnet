#!/usr/bin/env bash

# Based on https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/loca/data.sh

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


stage=4
stop_stage=100000
data_url=www.openslr.org/resources/12
train_set="train_all"
train_dev="dev"

# librispeech subsets that are used to generate train/dev data
train_subsets="train_clean_100 train_clean_360 train_other_500"
dev_subsets="dev_clean dev_other"
use_external_data=false


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

# NOTE: assuming data is present
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  if [ ! -e "${LIBRISPEECH}/LibriSpeech/LICENSE.TXT" ]; then
    log "stage 1: Require librispeech data to be present at ${LIBRISPEECH}"
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/prep-librispeech.sh ${LIBRISPEECH}/LibriSpeech/${part} data/${part//-/_}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: combine all training and development sets"
    train_subsets=$(sed 's/[^ ]* */data\/&/g' <<< ${train_subsets})
    dev_subsets=$(sed 's/[^ ]* */data\/&/g' <<< ${dev_subsets})
    log "Using ${train_subsets} for the train set and ${dev_subsets} for the dev set"
    utils/combine_data.sh --extra_files utt2num_frames data/${train_set} ${train_subsets}
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev} ${dev_subsets}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  align_dir=local/librispeech-aligned
  # NOTE: generate phone-level transcripts like so:
  # pushd $align_dir
  #  ./align-librispeech.sh
  # popd

  cp $align_dir/text data/${train_set}/text
  utils/fix_data_dir.sh data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
