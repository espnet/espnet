#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of FBANK features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

test=data-fbank/test
train=data-fbank/train

test_original=data/test
train_original=data/train

gmm=exp/tri3

stage=0
. utils/parse_options.sh || exit 1;

# Make the FBANK features
[ ! -e $test ] && if [ $stage -le 0 ]; then
  # Dev set
  utils/copy_data_dir.sh $test_original $test || exit 1; rm $test/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $test $test/log $test/data || exit 1;
  steps/compute_cmvn_stats.sh $test $test/log $test/data || exit 1;
  # Training set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd --max-jobs-run 10" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (117h, regular DNN)
  dir=exp/dnn4d-fbank_pretrain-dbn
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      $train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4d-fbank_pretrain-dbn_dnn
  ali=${gmm}_ali
  feature_transform=exp/dnn4d-fbank_pretrain-dbn/final.feature_transform
  dbn=exp/dnn4d-fbank_pretrain-dbn/6.dbn
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 11 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph $test $dir/decode_test || exit 1;
fi


# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
dir=exp/dnn4d-fbank_pretrain-dbn_dnn_smbr
srcdir=exp/dnn4d-fbank_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 11 --cmd "$train_cmd" \
    $train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 11 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 6 iterations of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2 3 4 5 6; do
    steps/nnet/decode.sh --nj 11 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmm/graph $test $dir/decode_test_it${ITER} || exit 1
  done 
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
