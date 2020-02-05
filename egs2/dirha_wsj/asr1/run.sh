#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

mic=Beam_Circular_Array

local_data_opts="--mic "${mic}" "

stage=9
stop_stage=9
nj=16
decode_asr_model=eval.loss.best.pth   # espnet 1 recog_model=model.acc.best

train_set=train_si284_$mic
dev_set=dirha_sim_$mic
eval_set=dirha_real_$mic

# config files
#preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
asr_config=conf/train.yaml
decode_config=conf/decode.yaml

lm_config=conf/lm.yaml
use_wordlm=false # not supported yet
word_vocab_size=65000

#--local_data_opts "${local_data_opts}" \
./asr.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nj ${nj} \
    --decode_asr_model ${decode_asr_model} \
    --token_type char \
    --feats_type fbank_pitch \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_set}" \
    --srctexts "data/${train_set}/text data/local/other_text/text" "$@"
