#!/bin/bash

# This script prepares data and trains + decodes an ASR system.

# initialization PATH
. ./path.sh  || die "path.sh expected";
# initialization commands
. ./cmd.sh

[ ! -L "steps" ] && ln -s ../../wsj/s5/steps

[ ! -L "utils" ] && ln -s ../../wsj/s5/utils

###############################################################
#                   Configuring the ASR pipeline
###############################################################
stage=0    # from which stage should this script start
stop_stage=100
nj=48        # number of parallel jobs to run during training;
decode_nj=24    # number of parallel jobs to run for test during decoding
. ./utils/parse_options.sh
# the above two parameters are bounded by the number of speakers in each set and number of CPUs available
# decrease the number of parallel jobs if needed to conserve RAM or CPU usage
###############################################################

# Stage 1: Prepares the train/test data. Prepares the dictionary and the
# language model.
if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Preparing data and training language models"
  local/prepare_data.sh train test
  local/prepare_dict.sh
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  local/prepare_lm.sh
fi

# Feature extraction
# Stage 2: MFCC feature extraction + mean-variance normalization
if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   for x in train test; do
      steps/make_mfcc.sh --nj "$nj" --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
   done
fi

# Stage 3: Training and decoding monophone acoustic models
if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  ### Monophone
    echo "mono training"
	steps/train_mono.sh --nj "$nj" --cmd "$train_cmd" data/train data/lang exp/mono
    echo "mono training done"
    (
    echo "mono decoding on test"
    utils/mkgraph.sh data/lang exp/mono exp/mono/graph
  
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
      exp/mono/graph data/test exp/mono/decode_test
    echo "mono decoding on test done"
    ) &
fi

# Stage 4: Training tied-state triphone acoustic models
if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  ### Triphone
    echo "tri1 training"
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
       data/train data/lang exp/mono exp/mono_ali
	steps/train_deltas.sh --boost-silence 1.25  --cmd "$train_cmd"  \
	   2000 20000 data/train data/lang exp/mono_ali exp/tri1
    echo "tri1 training done"
    (
    echo "tri1 decoding on test"
    utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph
  
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
      exp/tri1/graph data/test exp/tri1/decode_test
    echo "tri1 decoding on test done."
    ) &
fi

# Stage 5: Train an LDA+MLLT system.
if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "tri2b training"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train data/lang exp/tri1_ali exp/tri2b

  echo "tri2b training done"
  # decode using the LDA+MLLT model
  (
    echo "tri2b decoding on test"
    utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
  
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
      exp/tri2b/graph data/test exp/tri2b/decode_test
    echo "tri2b decoding on test done."
  ) &
fi

# Stage 6: Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "tri3b training"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2b exp/tri2b_ali

  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang exp/tri2b_ali exp/tri3b
  echo "tri3b training done"

  # decode using the tri3b model
  (
    echo "tri3b decoding on test"
    utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
      exp/tri3b/graph data/test exp/tri3b/decode_test
    echo "tri3b decoding on test done."
  )&
fi

# Stage 7: Train tri4b, which is LDA+MLLT+SAT
if [ $stage -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "tri4b training"
  # Align utts in the full training set using the tri3b model
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang \
    exp/tri3b exp/tri3b_ali

  # train another LDA+MLLT+SAT system on the entire training set
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
    data/train data/lang \
    exp/tri3b_ali exp/tri4b
  echo "tri4b training done"

  # decode using the tri4b model
  (
    echo "tri4b decoding on test"
    utils/mkgraph.sh data/lang exp/tri4b exp/tri4b/graph
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
      exp/tri4b/graph data/test exp/tri4b/decode_test
    echo "tri4b decoding on test done."
  )&
fi

# Train a chain model
if [ $stage -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  local/chain/run_tdnn.sh --stage 0 --nj $nj --decode_nj $decode_nj
fi

wait;
#score
# Computing the best WERs
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
for x in exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
