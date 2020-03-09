#!/bin/bash
# Set bash to  'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e 
set -u
set -o pipefail

ngpu=$1

#tts_config=conf/train_transformer.v3.yaml
tts_config=conf/train_fastspeech.v3.yaml
decode_config=conf/decode.yaml

teacher_model_path=exp/tts_train_tacotron2.v3_fbank/valid.loss.best.pth
teacher_model_config=exp/tts_train_tacotron2.v3_fbank/config.yaml

./tts.sh \
  --ngpu ${ngpu} \
  --audio_format wav \
  --feats_type fbank \
  --fs 22050 \
  --train_config ${tts_config} \
  --decode_config ${decode_config} \
  --trans_type phn \
  --srctexts  "data/phn_train_nodev/text" \
  --griffin_lim_iters 100 \
  --stage 6 \
  --stop_stage 6 \
  --gpu_decode true \
  #--teacher_model_path ${teacher_model_path} \
  #--teacher_model_config ${teacher_model_config} \
  #--tag test_plot_att


