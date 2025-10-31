#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

train_set=train
valid_set=dev
test_sets="test_aishell test_cv test_doreco test_fleurs test_kazakh test_librispeech test_mls_dutch test_mls_french test_mls_german test_mls_italian test_mls_polish test_mls_portuguese test_mls_spanish test_tamil"

nbpe=40000
# s2t config based on owsm v3.1 small, and some modifications from owsm v4
s2t_config=conf/tuning/train_s2t_ebf_conv2d_size768_e9_d9_piecewise_lr5e-4_warmup60k_flashattn.yaml
inference_config=conf/decode_s2t_pr.yaml


# Specify nodelist based on your environment with --host "$nodelist"
# nodelist=$(scontrol show hostnames $SLURM_JOB_NODELIST)
# nodelist=$(echo $nodelist | tr ' ' ',')

./s2t.sh \
    --stage 1 \
    --stop_stage 11 \
    --use_lm false \
    --ngpu 4 \
    --nj 64 \
    --gpu_inference true \
    --inference_nj 32 \
    --num_splits_s2t 1 \
    --max_wav_duration 20 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 35000000 \
    --bpe_largecorpus true \
    --local_data_opts "--stage 1 --stop_stage 1" \
    --post_process_local_data_opts "--stage 2 --stop_stage 2" \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/bpe_nlsyms.txt \
    --nlsyms_txt data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
