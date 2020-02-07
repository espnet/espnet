#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

mic=L1C     # Beam_Circular_Array Beam_Linear_Array KA6 L1C

local_data_opts="--mic ${mic}"

stage=1
stop_stage=11
nj=16
decode_asr_model=valid.loss.best.pth 

train_set=train_si284_$mic
dev_set=dirha_sim_$mic
eval_set=dirha_real_$mic

# config files
#preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
asr_config=conf/train.yaml
decode_config=conf/decode.yaml

lm_config=conf/lm.yaml
use_word_lm=false
word_vocab_size=65000

./asr.sh                                        \
    --stage ${stage}                            \
    --stop_stage ${stop_stage}                  \
    --nj ${nj}                                  \
    --decode_asr_model ${decode_asr_model}      \
    --token_type char                           \
    --feats_type fbank_pitch                    \
    --asr_config "${asr_config}"                \
    --decode_config "${decode_config}"          \
    --lm_config "${lm_config}"                  \
    --use_word_lm ${use_word_lm}                \
    --word_vocab_size ${word_vocab_size}        \
    --train_set "${train_set}"                  \
    --dev_set "${dev_set}"                      \
    --eval_sets "${eval_set}"                   \
    --local_data_opts "${local_data_opts}"      \
    --srctexts "data/${train_set}/text data/local/other_text/text" "$@"
