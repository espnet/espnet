#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# bert

./run_entailment.sh "$@" &
sleep 2s

./run_aqa_yn.sh "$@" &
sleep 2s

./run_aqa_open.sh "$@" &
sleep 2s

# clap
./run_entailment.sh \
    --cls_config conf/beats_clap_multimodal_cls.yaml \
    --hugging_face_model_name_or_path laion/clap-htsat-unfused \
    --expdir ./exp/cle_clap "$@" &
sleep 2s

./run_aqa_yn.sh \
    --cls_config conf/beats_clap_multimodal_cls.yaml \
    --hugging_face_model_name_or_path laion/clap-htsat-unfused \
    --expdir ./exp/aqa_yn_clap "$@" &
sleep 2s

./run_aqa_open.sh \
    --cls_config conf/beats_clap_multimodal_cls.yaml \
    --hugging_face_model_name_or_path laion/clap-htsat-unfused \
    --expdir ./exp/aqa_open_clap "$@" &

wait