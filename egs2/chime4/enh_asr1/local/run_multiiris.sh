#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


enh_train_set=tr05_simu_isolated_6ch_track
enh_valid_set=dt05_simu_isolated_6ch_track
enh_test_sets="et05_simu_isolated_6ch_track"
asr_train_set=tr05_multi_noisy_si284
asr_valid_set=dt05_multi_isolated_1ch_track
asr_test_sets="\
dt05_simu_beamformit_5mics dt05_real_beamformit_5mics \
et05_simu_beamformit_5mics et05_real_beamformit_5mics \
"
enh_asr_train_set=tr05_multi_isolated_6ch_track
enh_asr_valid_set=dt05_multi_isolated_6ch_track
enh_asr_test_sets="\
dt05_simu_isolated_6ch_track dt05_real_isolated_6ch_track \
et05_simu_isolated_6ch_track et05_real_isolated_6ch_track \
"


enh_config=../enh1/conf/tuning/train_enh_beamformer_wpd_ci_sdr_shorttap.yaml
asr_config=../asr1/conf/tuning/train_asr_conformer_wavlm2.yaml
lm_config=../asr1/conf/train_lm_transformer.yaml
enh_asr_config=../enh_asr1/conf/tuning/train_enh_asr_wpd_init_noenhloss_wavlm_conformer.yaml

lm_exp=../asr1/exp/lm_train_lm_transformer_en_char
inference_config=../enh_asr1/conf/decode_asr_transformer_largelm.yaml
inference_enh_asr_model=valid.acc.ave_10best.pth

ref_channel=4
use_word_lm=false
word_vocab_size=65000
extra_annotations=/espnet/datasets/CHiME4/CHiME3/data/annotations


# Pretraining of enhancement model
./run_enh_pretraining.sh \
    --stage 1 \
    --stop_stage 6 \
    --train_set "${enh_train_set}" \
    --valid_set "${enh_valid_set}" \
    --test_sets "${enh_test_sets}" \
    --enh_config "${enh_config}" \
    --extra_annotations "${extra_annotations}" \
    --ref_channel "${ref_channel}"


# Pretraining of ASR model and Language model
./run_asr_pretraining.sh \
    --stage 1 \
    --stop_stage 11 \
    --train_set "${asr_train_set}" \
    --valid_set "${asr_valid_set}" \
    --test_sets "${asr_test_sets}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --use_word_lm "${use_word_lm}" \
    --word_vocab_size "${word_vocab_size}"


# Preparing for joint finetuning
./run_enh_asr_finetuning.sh \
    --stage 1 \
    --stop_stage 5 \
    --train_set "${enh_asr_train_set}" \
    --valid_set "${enh_asr_valid_set}" \
    --test_sets "${enh_asr_test_sets}" \
    --enh_asr_config "${enh_asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --use_word_lm "${use_word_lm}" \
    --word_vocab_size "${word_vocab_size}" \
    --extra_annotations "${extra_annotations}" \
    --ref_channel "${ref_channel}" \
    --lm_exp "${lm_exp}" \
    --inference_enh_asr_model "${inference_enh_asr_model}"


# Copy tokens used in ASR model and language model
cp ../../asr1/data/en_token_list/char/tokens.txt ../../enh_asr1/data/en_token_list/char/tokens.txt


# Fintuning and evaluating entire model
./run_enh_asr_finetuning.sh \
    --stage 10 \
    --stop_stage 15 \
    --train_set "${enh_asr_train_set}" \
    --valid_set "${enh_asr_valid_set}" \
    --test_sets "${enh_asr_test_sets}" \
    --enh_asr_config "${enh_asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --use_word_lm "${use_word_lm}" \
    --word_vocab_size "${word_vocab_size}" \
    --extra_annotations "${extra_annotations}" \
    --ref_channel "${ref_channel}" \
    --lm_exp "${lm_exp}" \
    --inference_enh_asr_model "${inference_enh_asr_model}"
