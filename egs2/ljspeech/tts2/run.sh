#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr_no_dev
train_dev=dev
eval_set=eval1

feats_type=raw
s3prl_upstream_name=hubert
feature_layer=6
feature_num_clusters=500

# put your vocoder file and vocoder config file here
# vocoder can be trained from
# https://github.com/kan-bayashi/ParallelWaveGAN
vocoder_file=vocoder/vocoder.pkl

# duration information
teacher_dumpdir=teacher_dumpdir

./tts2.sh \
    --fs 24000 --n_shift 480 --n_fft 1024 --win_length 1024 \
    --s3prl_upstream_name ${s3prl_upstream_name} \
    --feature_layer ${feature_layer} \
    --feature_num_clusters ${feature_num_clusters} \
    --lang en \
    --teacher_dumpdir ${teacher_dumpdir} \
    --train_config conf/train_fastspeech2.yaml \
    --train_set ${train_set} \
    --valid_set ${train_dev} \
    --test_sets ${eval_set} \
    --write_collected_feats true \
    --vocoder_file ${vocoder_file} \
    --feats_type ${feats_type} \
    --srctexts "data/tr_no_dev/text" "$@"

