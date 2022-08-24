#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

outdir=aishell4_simu
sample_rate=16k


train_set="train"
valid_set="dev"
test_sets="test"

./enh.sh \
    --audio_format wav \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --lang zh \
    --ngpu 1 \
    --local_data_opts "--outdir ${outdir}" \
    --enh_config conf/tuning/train_enh_beamformer_no_wpe.yaml \
    "$@"
