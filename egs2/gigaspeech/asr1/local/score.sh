#!/usr/bin/env bash

set -e
set -u
set -o pipefail

lm_exp=
inference_tag=
inference_config=
inference_args=
use_lm=
inference_lm=valid.loss.ave.pth
inference_asr_model=valid.acc.ave.pth
test_sets="dev test"

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./utils/parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
    echo "Usage:  <asr_exp> "
    echo "Remove conversational fillers from both hypothese and reference text"
    echo "and resore for normalization"
    exit 1
fi

asr_exp=$1

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

if [ -n "${test_sets}" ]; then
    for dset in ${test_sets}; do
        _dir="${asr_exp}/${inference_tag}/${dset}"
        for _type in cer wer ter; do
            _scoredir="${_dir}/score_${_type}"
            python GigaSpeech/utils/gigaspeech_scoring.py ${_scoredir}/ref.trn ${_scoredir}/hyp.trn ${_scoredir}
        done
    done
fi
