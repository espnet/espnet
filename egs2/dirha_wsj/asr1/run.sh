#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
valid_set=dirha_sim_Livingroom_Circular_Array_Beam_Circular_Array
test_sets=
test_sets+=" dirha_real_Kitchen_Circular_Array_KA1"
test_sets+=" dirha_real_Kitchen_Circular_Array_KA2"
test_sets+=" dirha_real_Kitchen_Circular_Array_KA3"
test_sets+=" dirha_real_Kitchen_Circular_Array_KA4"
test_sets+=" dirha_real_Kitchen_Circular_Array_KA5"
test_sets+=" dirha_real_Kitchen_Circular_Array_KA6"
test_sets+=" dirha_real_Livingroom_Circular_Array_Beam_Circular_Array"
test_sets+=" dirha_real_Livingroom_Circular_Array_LA1"
test_sets+=" dirha_real_Livingroom_Circular_Array_LA2"
test_sets+=" dirha_real_Livingroom_Circular_Array_LA3"
test_sets+=" dirha_real_Livingroom_Circular_Array_LA4"
test_sets+=" dirha_real_Livingroom_Circular_Array_LA5"
test_sets+=" dirha_real_Livingroom_Circular_Array_LA6"
test_sets+=" dirha_real_Livingroom_Linear_Array_Beam_Linear_Array"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD02"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD03"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD04"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD05"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD06"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD07"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD08"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD09"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD10"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD11"
test_sets+=" dirha_real_Livingroom_Linear_Array_LD12"
test_sets+=" dirha_real_Livingroom_Wall_L1C"
test_sets+=" dirha_real_Livingroom_Wall_L1L"
test_sets+=" dirha_real_Livingroom_Wall_L1R"
test_sets+=" dirha_real_Livingroom_Wall_L2L"
test_sets+=" dirha_real_Livingroom_Wall_L2R"
test_sets+=" dirha_real_Livingroom_Wall_L3L"
test_sets+=" dirha_real_Livingroom_Wall_L3R"
test_sets+=" dirha_real_Livingroom_Wall_L4L"
test_sets+=" dirha_real_Livingroom_Wall_L4R"
test_sets+=" dirha_sim_Kitchen_Circular_Array_KA1"
test_sets+=" dirha_sim_Kitchen_Circular_Array_KA2"
test_sets+=" dirha_sim_Kitchen_Circular_Array_KA3"
test_sets+=" dirha_sim_Kitchen_Circular_Array_KA4"
test_sets+=" dirha_sim_Kitchen_Circular_Array_KA5"
test_sets+=" dirha_sim_Kitchen_Circular_Array_KA6"
test_sets+=" dirha_sim_Livingroom_Circular_Array_Beam_Circular_Array"
test_sets+=" dirha_sim_Livingroom_Circular_Array_LA1"
test_sets+=" dirha_sim_Livingroom_Circular_Array_LA2"
test_sets+=" dirha_sim_Livingroom_Circular_Array_LA3"
test_sets+=" dirha_sim_Livingroom_Circular_Array_LA4"
test_sets+=" dirha_sim_Livingroom_Circular_Array_LA5"
test_sets+=" dirha_sim_Livingroom_Circular_Array_LA6"
test_sets+=" dirha_sim_Livingroom_Linear_Array_Beam_Linear_Array"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD02"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD03"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD04"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD05"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD06"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD07"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD08"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD09"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD10"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD11"
test_sets+=" dirha_sim_Livingroom_Linear_Array_LD12"
test_sets+=" dirha_sim_Livingroom_Wall_L1C"
test_sets+=" dirha_sim_Livingroom_Wall_L1L"
test_sets+=" dirha_sim_Livingroom_Wall_L1R"
test_sets+=" dirha_sim_Livingroom_Wall_L2L"
test_sets+=" dirha_sim_Livingroom_Wall_L2R"
test_sets+=" dirha_sim_Livingroom_Wall_L3L"
test_sets+=" dirha_sim_Livingroom_Wall_L3R"
test_sets+=" dirha_sim_Livingroom_Wall_L4L"
test_sets+=" dirha_sim_Livingroom_Wall_L4R"

# config files
asr_config=conf/tuning/train_asr_transformer_cmvn.yaml
lm_config=conf/tuning/train_lm_transformer.yaml
inference_config=conf/decode.yaml

use_word_lm=false
word_vocab_size=65000

./asr.sh                                        \
    --lang en \
    --ngpu 4 \
    --audio_format flac \
    --nlsyms_txt data/nlsyms.txt                \
    --token_type char                           \
    --feats_type raw                    \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}"                \
    --inference_config "${inference_config}"          \
    --lm_config "${lm_config}"                  \
    --use_word_lm ${use_word_lm}                \
    --word_vocab_size ${word_vocab_size}        \
    --train_set "${train_set}"                  \
    --valid_set "${valid_set}"                  \
    --test_sets "${test_sets}"                  \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" "$@"
