#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr_no_dev
train_dev=dev
eval_set=eval1

# put your vocoder file and vocoder config file here
# vocoder can be trained from
# https://github.com/kan-bayashi/ParallelWaveGAN
vocoder_file=vocoder/vocoder.pkl

# duration information
teacher_dumpdir=teacher_dumpdir

./tts2.sh \
    --nj 16 \
    --inference_nj 16 \
    --fs 16000 --n_shift 320 --n_fft 1280 \
    --s3prl_upstream_name hubert \
    --feature_layer 6 \
    --feature_num_clusters 500 \
    --lang en \
    --teacher_dumpdir ${teacher_dumpdir} \
    --train_config conf/train_fastspeech2.yaml \
    --train_set ${train_set} \
    --valid_set ${train_dev} \
    --test_sets ${eval_set} \
    --write_collected_feats true \
    --vocoder_file ${vocoder_file} \
    --srctexts "data/tr_no_dev/text" "$@"
