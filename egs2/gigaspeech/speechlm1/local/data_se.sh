#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

train_set=
valid_set=
test_sets=

noise_scp=/scratch/bbjs/jtian1/espnet_speechlm_data/egs2/librispeech/speechlm1/audioset/wav.scp
rir_scp=/scratch/bbjs/jtian1/espnet_speechlm_data/egs2/librispeech/speechlm1/rir/rir.scp
nj=16

log "$0 $*"
. utils/parse_options.sh

. ./path.sh || exit 1
. ./cmd.sh || exit 1

for dset in ${train_set} ${valid_set} ${test_sets}; do
    log "Generate Speech Enhancement and Separation data for data/${dset}"

    mkdir -p data/${dset}_enh data/${dset}_sep

    total_lines=$(wc -l < data/${dset}/wav.scp)
    half_lines=$((total_lines / 2))

    shuf data/${dset}/wav.scp > data/${dset}_enh/shuffled_wav.scp
    head -n $half_lines          data/${dset}_enh/shuffled_wav.scp > data/${dset}_enh/wav.scp
    tail -n +$((half_lines + 1)) data/${dset}_enh/shuffled_wav.scp > data/${dset}_sep/wav.scp
    rm data/${dset}_enh/shuffled_wav.scp

    # enhancement
    log "Start generating Speech Enhancement Data"
    bash scripts/audio/dump_noisy_speech.sh \
        --input_scp data/${dset}_enh/wav.scp \
        --noise_scp ${noise_scp} \
        --rir_scp ${rir_scp} \
        --output_dir data/${dset}_enh \
        --nj ${nj}

    separation
    log "Start generating Speech Separation Data"
    bash scripts/audio/dump_noisy_speech.sh \
        --input_scp data/${dset}_sep/wav.scp \
        --noise_scp data/${dset}_enh/spk1.scp \
        --output_dir data/${dset}_sep \
        --nj ${nj} \
        --rir_apply_prob 0.0

    mkdir -p data/${dset}_enh_sep
    cat data/${dset}_enh/wav.scp data/${dset}_sep/wav.scp > data/${dset}_enh_sep/wav.scp
    cat data/${dset}_enh/spk1.scp data/${dset}_sep/spk1.scp > data/${dset}_enh_sep/spk1.scp
done
