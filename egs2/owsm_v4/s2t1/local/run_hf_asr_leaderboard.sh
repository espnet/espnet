#!/usr/bin/env bash
# Sample a few Open ASR Leaderboard test sets (~100 utterances each) and
# measure WER and RTFx of an OWSM checkpoint on them, on CPU by default.
#
# Usage:
#   local/run_hf_asr_leaderboard.sh [--model_tag espnet/owsm_v4_base_102M] \
#       [--n 100] [--batch_size 8] [--maxlenratio 0.0] [--sort true] [--threads 4]
#
# The dataset is gated: accept the terms at
# https://huggingface.co/datasets/hf-audio/open-asr-leaderboard and log in
# with `huggingface-cli login` (or set HF_TOKEN) before running.

set -euo pipefail

model_tag=espnet/owsm_v4_base_102M
n=100
seed=0
batch_size=8
beam_size=5
maxlenratio=0.0
sort=true
threads=4
quantize=false
device=cpu
dtype=float32
outdir=exp/hf_asr_leaderboard
# "config:split" pairs of hf-audio/open-asr-leaderboard
sets="librispeech:test.clean librispeech:test.other ami_cleaned:test voxpopuli_cleaned_aa:test earnings22:test"

. utils/parse_options.sh
. ./path.sh

mkdir -p "${outdir}"
summary="${outdir}/summary.md"
{
    echo "Model: ${model_tag}, n=${n} per set, seed=${seed}, device=${device}, threads=${threads}"
    echo
    echo "| set | model | decoding | WER (%) | RTFx | s/utt |"
    echo "| --- | --- | --- | ---: | ---: | ---: |"
} > "${summary}"

for pair in ${sets}; do
    dataset=${pair%%:*}
    split=${pair#*:}
    name=$(echo "lb_${dataset}_${split}" | tr -c 'A-Za-z0-9_\n' '_')
    data="data/${name}"
    if [ ! -f "${data}/wav.scp" ]; then
        echo "=== preparing ${data}"
        python3 local/prepare_hf_asr_leaderboard.py \
            --dataset "${dataset}" --split "${split}" --n "${n}" --seed "${seed}" \
            --out "${data}"
    fi
    echo "=== decoding ${data}"
    python3 local/eval_hf_asr_leaderboard.py \
        --data "${data}" --model_tag "${model_tag}" --device "${device}" \
        --dtype "${dtype}" --batch_size "${batch_size}" --beam_size "${beam_size}" \
        --maxlenratio "${maxlenratio}" --sort "${sort}" --threads "${threads}" \
        --quantize "${quantize}" \
        --out "${outdir}/${name}__${model_tag##*/}__bs${batch_size}_beam${beam_size}.json" \
        | tail -n 1 >> "${summary}"
done

echo
cat "${summary}"
