#!/bin/bash
# Set bash to  'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e 
set -u
set -o pipefail


tts_config=conf/train_transformer.v3.yaml
decode_config=conf/decode.yaml


./tts.sh \
  --ngpu 1 \
  --audio_format wav \
  --feats_type fbank \
  --fs 22050 \
  --train_config ${tts_config} \
  --decode_config ${decode_config} \
  --trans_type phn \
  --srctexts  "data/phn_train_nodev/text" \
  --griffin_lim_iters 100 \
  --stage 5 \
  --stop_stage 5 \
  #--tag test_plot_att


