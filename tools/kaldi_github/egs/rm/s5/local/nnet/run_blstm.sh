#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a BLSTM network on FBANK features.
# The initial BLSTM code comes from Ni Chongjia (I2R), thanks!

# We use multi-stream training, while the BPTT is done over whole
# utterances with similar length (selection done with C++ class MatrixBuffer).

# Note: With DNNs in RM, the optimal LMWT is 2-6. Don't be tempted to try acwt's like 0.2,
# the value 0.1 is better both for decoding and sMBR.

. ./cmd.sh
. ./path.sh

dev=data-fbank/test
train=data-fbank/train

dev_original=data/test
train_original=data/train

gmm=exp/tri3b

stage=0
. utils/parse_options.sh || exit 1;

set -eu

# Make the FBANK features
[ ! -e $dev ] && if [ $stage -le 0 ]; then
  # Dev set
  utils/copy_data_dir.sh $dev_original $dev || exit 1; rm $dev/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $dev $dev/log $dev/data || exit 1;
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;
  # Training set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd --max-jobs-run 10" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi

if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/blstm4i
  ali=${gmm}_ali

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --network-type blstm --learn-rate 0.00004 \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --feat-type plain --splice 0 \
      --scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
      --train-tool "nnet-train-multistream-perutt" \
      --train-tool-opts "--num-streams=10 --max-frames=15000" \
      --proto-opts "--cell-dim 320 --proj-dim 200 --num-layers 2" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $dev $dir/decode || exit 1;
fi

# TODO : sequence training,

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
