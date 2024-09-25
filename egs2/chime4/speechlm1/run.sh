#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16000
extra_annotations=

train_set=tr05_simu_isolated_1ch_track
valid_set=dt05_simu_isolated_1ch_track
test_sets=et05_simu_isolated_1ch_track

codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"

./speechlm.sh \
    --task "se" \
    --data_name "chime4" \
    --fs $sample_rate \
    --nj 16 \
    --inference_nj 16 \
    --audio_format "wav.ark" \
    --train_config conf/tuning/train_delay_se.yaml \
    --inference_config conf/tuning/decode_se.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_model latest.pth \
    --local_data_opts "--extra-annotations ${extra_annotations} --stage 1 --stop-stage 2" \
    $codec_opts \
    "$@"
