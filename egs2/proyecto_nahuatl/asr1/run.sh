#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

prefix=

trans_prefix_Zacatlan=${prefix}/Zacatlan-Tepetzintla/Transcripciones-finales
trans_prefix_Tequila=${prefix}/Tequila-Zongolica/Transcripciones-finales
trans_prefix_Hidalgo=${prefix}/Hidalgo-Transcripciones/Transcripciones-Finales

audio_prefix_Zacatlan=${prefix}/Zacatlan-Tepetzintla/Grabaciones_Por-dia
audio_prefix_Tequila=${prefix}/Tequila-Zongolica/Grabaciones
audio_prefix_Hidalgo=${prefix}/Hidalgo-Grabaciones

asr_config=conf/train_asr_s3prl.yaml
inference_config=conf/decode_asr_ctc.yaml

# specify audio directory for processing the audio
audio_dir=downloads/wav

./asr.sh \
    --asr_config "${asr_config}" \
    --inference_config "${conf/decode_asr_ctc.yaml}" \
    --audio_dir "${audio_dir}"
    --lm_config conf/train_lm.yaml \
    --feats_normalize utterance_mvn \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --train_set train \
    --valid_set dev \
    --trans_prefix_Zacatlan "${trans_prefix_Zacatlan}" \
    --trans_prefix_Tequila "${trans_prefix_Tequila}" \
    --trans_prefix_Hidalgo "${trans_prefix_Hidalgo}" \
    --audio_prefix_Zacatlan "${audio_prefix_Zacatlan}" \
    --audio_prefix_Tequila "${audio_prefix_Tequila}" \
    --audio_prefix_Hidalgo "${audio_prefix_Hidalgo}" \
    --test_sets "test/Zacatlan test/Tequila test/Hidalgo" \
    --bpe_train_text "data/train/text" \
    --lm_train_text "data/train/text" "$@"
