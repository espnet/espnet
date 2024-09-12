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

. ./path.sh
. ./cmd.sh

stage=1
stop_stage=100

# NOTE(jiatong): placeholder, not used
nj=
inference_nj=
gpu_inference=
nbest=
key_file=

# Evaluation dir
gen_dir=
ref_dir=
expdir=exp

token_type=word
nlsyms_txt=none
cleaner=none
hyp_cleaner=none
lang=de
case=lc

python=python3

log "$0 $*"
. utils/parse_options.sh

scoredir=${gen_dir}/scoring/eval_bleu
mkdir -p ${scoredir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Prepare scoring
    paste \
        <(<"${ref_dir}/text" \
            ${python} -m espnet2.bin.tokenize_text  \
                -f 2- --input - --output - \
                --token_type word \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --remove_non_linguistic_symbols true \
                --cleaner "${cleaner}" \
            ) \
        >"${scoredir}/ref.trn" 

    paste \
        <(<"${gen_dir}/text" \
            ${python} -m espnet2.bin.tokenize_text  \
                -f 2- --input - --output - \
                --token_type word \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --remove_non_linguistic_symbols true \
                --cleaner "${hyp_cleaner}" \
            ) \
        >"${scoredir}/hyp.trn"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Detokenize
    detokenizer.perl -l ${lang} -q < "${scoredir}/ref.trn" > "${scoredir}/ref.trn.detok"
    detokenizer.perl -l ${lang} -q < "${scoredir}/hyp.trn" > "${scoredir}/hyp.trn.detok"

    lowercase.perl < "${scoredir}/ref.trn.detok" > "${scoredir}/ref.trn.detok.lc"
    lowercase.perl < "${scoredir}/hyp.trn.detok" > "${scoredir}/hyp.trn.detok.lc"

    scripts/utils/remove_punctuation.pl < "${scoredir}/ref.trn.detok.lc" > "${scoredir}/ref.trn.detok.lc.rm"
    scripts/utils/remove_punctuation.pl < "${scoredir}/hyp.trn.detok.lc" > "${scoredir}/hyp.trn.detok.lc.rm"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Scoring
    if [ ${case} = "tc" ]; then
        echo "Case sensitive BLEU result (single-reference)" > ${scoredir}/result.tc.txt
        sacrebleu "${scoredir}/ref.trn.detok" \
                  -i "${scoredir}/hyp.trn.detok" \
                  -m bleu chrf ter \
                  >> ${scoredir}/result.tc.txt

        log "Write a case-sensitive BLEU (single-reference) result in ${scoredir}/result.tc.txt"
    fi

    # Always conduct scoring for case insensitive settings
        sacrebleu "${scoredir}/ref.trn.detok.lc.rm" \
                  -i "${scoredir}/hyp.trn.detok.lc.rm" \
                  -m bleu chrf ter \
                  >> ${scoredir}/result.lc.txt

    log "Write a case-sensitive BLEU (single-reference) result in ${scoredir}/result.lc.txt"

    # Show results in Markdown syntax
    scripts/utils/show_translation_result.sh --case ${case} ${expdir} > ${expdir}/Translation_RESULT.md
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
