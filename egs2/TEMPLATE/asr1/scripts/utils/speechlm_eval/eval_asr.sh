#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
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

# . ./path.sh
. ./cmd.sh

stage=1
stop_stage=100
nj=8
inference_nj=8
gpu_inference=true
nbest=1

gen_dir=
ref_dir=
key_file=

# wer options
cleaner=whisper_en
hyp_cleaner=whisper_en
scoring_metrics="cer wer"
nlsyms_txt=none

python=python3

log "$0 $*"
. utils/parse_options.sh

mkdir -p ${gen_dir}/scoring

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Scoring"
    for _type in ${scoring_metrics}; do
        _scoredir="${gen_dir}/scoring/score_${_type}"
        mkdir -p "${_scoredir}"

        gen_text=${gen_dir}/text
        gt_text=${ref_dir}/text

        if [ "${_type}" = wer ]; then
            paste \
                <(<"${gt_text}" \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type word \
                          --non_linguistic_symbols "${nlsyms_txt}" \
                          --remove_non_linguistic_symbols true \
                          --cleaner "${cleaner}" \
                          ) \
                <(<"${gt_text}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/ref.trn"
            paste \
                <(<"${gen_text}"  \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type word \
                          --non_linguistic_symbols "${nlsyms_txt}" \
                          --remove_non_linguistic_symbols true \
                          --cleaner "${hyp_cleaner}" \
                          ) \
                <(<"${gen_text}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/hyp.trn"

        elif [ "${_type}" = cer ]; then
            paste \
                <(<"${gt_text}" \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type char \
                          --non_linguistic_symbols "${nlsyms_txt}" \
                          --remove_non_linguistic_symbols true \
                          --cleaner "${cleaner}" \
                          ) \
                <(<"${gt_text}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/ref.trn"
            paste \
                <(<"${gen_text}" \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type char \
                          --non_linguistic_symbols "${nlsyms_txt}" \
                          --remove_non_linguistic_symbols true \
                          --cleaner "${hyp_cleaner}" \
                          ) \
                <(<"${gen_text}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/hyp.trn"

        fi

        # Scoring
        sclite \
            -r "${_scoredir}/ref.trn" trn \
            -h "${_scoredir}/hyp.trn" trn \
            -i rm -o all stdout > "${_scoredir}/result.txt"

        log "Write ${_type} result in ${_scoredir}/result.txt"
        grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
