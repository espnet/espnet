#!/bin/bash
# Set bash to  'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e 
set -u
set -o pipefail


fs=22050
opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi
  
ngpu=$1

tts_config=conf/train_fastspeech.v3.yaml
decode_config=conf/decode.yaml

teacher_model_path=     # set this if use fastspeech model
                        # e.g, exp/tts_train_transformer.v3_fbank/valid.loss.best.pth
teacher_model_config=   # e.g, exp/tts_train_transformer.v3_fbank/config.yaml

trans_type=phn
train_set="${trans_type}_train_nodev"
dev_set="${trans_type}_dev"
eval_sets="${trans_type}_eval"
./tts.sh \
    --ngpu ${ngpu} \
    --audio_format wav \
    --feats_type fbank \
    --local_data_opts "--trans_type ${trans_type}" \
    --fs ${fs} \
    --train_set ${train_set} \
    --dev_set ${dev_set} \
    --eval_sets ${eval_sets} \
    --train_config ${tts_config} \
    --decode_config ${decode_config} \
    --srctexts  "data/phn_train_nodev/text" \
    --griffin_lim_iters 100 \
    --stage 6 \
    --stop_stage 6 \
    --gpu_decode true \
    --teacher_model_path ${teacher_model_path} \
    --teacher_model_config ${teacher_model_config} \
    ${opts} "$@"
