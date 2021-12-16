#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/tuning/train_asr_hubert_conformer.yaml
inference_config=conf/decode_asr_transformer.yaml

pretrained_hubert_asr=exp/exp_hubert_large_ll60k_weighted_perturb/asr_train_asr_conformer7_hubert_960hr_large_raw_en_bpe5000_sp/26epoch.pth
# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"
#finetune with hubert pretrained ASR e.g. train text: INTENT1 th _is a _trans _cript

./asr.sh                                               \
    --use_lm false                                     \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --feats_normalize utterance_mvn                    \
    --token_type bpe                                   \
    --nbpe 100                                         \
    --bpe_nlsyms DECREASEBRIGHTNESS,INCREASEBRIGHTNESS,SETLIGHTBRIGHTNESS,SETLIGHTCOLOR,SWITCHLIGHTOFF,SWITCHLIGHTON                                     \
    --bpe_train_text data/train/text                   \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_args " --init_param $pretrained_hubert_asr:::decoder.output_layer,decoder.embed.0,ctc.ctc_lo"    \
    --speed_perturb_factors "${speed_perturb_factors}" \
     "$@"
