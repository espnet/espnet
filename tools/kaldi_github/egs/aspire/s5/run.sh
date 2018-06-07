#!/bin/bash

#  ASpIRE submission, based on Fisher-english GMM-HMM system
# (March 2015)

# It's best to run the commands in this one by one.

. ./cmd.sh
. ./path.sh

mfccdir=`pwd`/mfcc

set -e

# Set this to somewhere where you want to put your aspire data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
aspire_data=/export/corpora/LDC/LDC2017S21/IARPA-ASpIRE-Dev-Sets-v2.0/data  # JHU

# the next command produces the data in local/train_all
local/fisher_data_prep.sh /export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
   /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13

local/fisher_prepare_dict.sh

utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

local/fisher_train_lms.sh  || exit 1;
local/fisher_create_test_lang.sh || exit 1;

# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.

utils/fix_data_dir.sh data/train_all

steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train_all exp/make_mfcc/train_all $mfccdir || exit 1;

utils/fix_data_dir.sh data/train_all
utils/validate_data_dir.sh data/train_all

# The dev and test sets are each about 3.3 hours long.  These are not carefully
# done; there may be some speaker overlap with each other and with the training
# set.  Note: in our LM-training setup we excluded the first 10k utterances (they
# were used for tuning but not for training), so the LM was not (directly) trained
# on either the dev or test sets.
utils/subset_data_dir.sh --first data/train_all 10000 data/dev_and_test
utils/subset_data_dir.sh --first data/dev_and_test 5000 data/dev
utils/subset_data_dir.sh --last data/dev_and_test 5000 data/test
rm -r data/dev_and_test

steps/compute_cmvn_stats.sh data/dev exp/make_mfcc/dev $mfccdir
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir

n=$[`cat data/train_all/segments | wc -l` - 10000]
utils/subset_data_dir.sh --last data/train_all $n data/train
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir


# Now-- there are 1.6 million utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.

utils/subset_data_dir.sh --shortest data/train 100000 data/train_100kshort
utils/subset_data_dir.sh  data/train_100kshort 10000 data/train_10k
utils/data/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup
utils/subset_data_dir.sh --speakers data/train 30000 data/train_30k
utils/subset_data_dir.sh --speakers data/train 100000 data/train_100k


# The next commands are not necessary for the scripts to run, but increase
# efficiency of data access by putting the mfcc's of the subset
# in a contiguous place in a file.
( . ./path.sh;
  # make sure mfccdir is defined as above..
  cp data/train_10k_nodup/feats.scp{,.bak}
  copy-feats scp:data/train_10k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_fish_10k_nodup.ark,$mfccdir/kaldi_fish_10k_nodup.scp \
  && cp $mfccdir/kaldi_fish_10k_nodup.scp data/train_10k_nodup/feats.scp
)
( . ./path.sh;
  # make sure mfccdir is defined as above..
  cp data/train_30k/feats.scp{,.bak}
  copy-feats scp:data/train_30k/feats.scp  ark,scp:$mfccdir/kaldi_fish_30k.ark,$mfccdir/kaldi_fish_30k.scp \
  && cp $mfccdir/kaldi_fish_30k.scp data/train_30k/feats.scp
)
( . ./path.sh;
  # make sure mfccdir is defined as above..
  cp data/train_100k/feats.scp{,.bak}
  copy-feats scp:data/train_100k/feats.scp  ark,scp:$mfccdir/kaldi_fish_100k.ark,$mfccdir/kaldi_fish_100k.scp \
  && cp $mfccdir/kaldi_fish_100k.scp data/train_100k/feats.scp
)

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_10k_nodup data/lang exp/mono0a

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k data/lang exp/mono0a_ali exp/tri1 || exit 1;


(utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1/graph data/dev exp/tri1/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k data/lang exp/tri1 exp/tri1_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k data/lang exp/tri1_ali exp/tri2 || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/dev exp/tri2/decode_dev || exit 1;
)&


steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 100k data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   5000 40000 data/train_100k data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/dev exp/tri3a/decode_dev || exit 1;
)&


# Next we'll use fMLLR and train with SAT (i.e. on
# fMLLR features)

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  5000 100000 data/train_100k data/lang exp/tri3a_ali  exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/dev exp/tri4a/decode_dev
)&


steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri4a exp/tri4a_ali || exit 1;


steps/train_sat.sh  --cmd "$train_cmd" \
  10000 300000 data/train data/lang exp/tri4a_ali  exp/tri5a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    exp/tri5a/graph data/dev exp/tri5a/decode_dev
)&

# build silprob lang directory
local/build_silprob.sh

local/multi_condition/aspire_data_prep.sh --aspire-data $aspire_data

# see local/{chain,nnet3}/* for nnet3 scripts

# train the neural network model
local/chain/run_tdnn.sh

local/chain/run_tdnn_lstm.sh
# %WER 22.9 | 2083 25834 | 81.6 12.0 6.4 4.5 22.9 70.7 | -0.546 | exp/chain/tdnn_lstm_1a/decode_dev_aspire_uniformsegmented_pp_fg/score_8/penalty_0.0/ctm.filt.filt.sys
# %WER 24.0 | 2083 25820 | 79.9 12.0 8.1 4.0 24.0 71.8 | -0.444 | exp/chain/tdnn_lstm_1a_online/decode_dev_aspire_uniformsegmented_v9_pp_fg/score_10/penalty_0.0/ctm.filt.filt.sys

# Train speech activity detection system using TDNN+Stats
local/run_asr_segmentation.sh

sad_nnet_dir=exp/segmentation_1a/tdnn_stats_asr_sad_1a
chain_dir=exp/chain/tdnn_lstm_1a

# %WER 22.9 | 2083 25821 | 81.9 11.1 7.0 4.9 22.9 71.8 | -0.488 | exp/chain/tdnn_lstm_1a/decode_dev_aspire_asr_sad_1a_pp_fg/score_10/penalty_0.25/ctm.filt.filt.sys
local/nnet3/segment_and_decode.sh --stage 1 --decode-num-jobs 30 --affix "" \
  --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 160 \
  --extra-left-context 50 --extra-right-context 0 \
  --extra-left-context-initial 0 --extra-right-context-final 0 \
  --sub-speaker-frames 6000 --max-count 75 \
  --decode-opts '--min-active 1000' \
  --sad-affix asr_sad_1a \
  --sad-opts "--extra-left-context 79 --extra-right-context 21 --frames-per-chunk 150 --extra-left-context-initial 0 --extra-right-context-final 0 --acwt 0.3" \
  --sad-graph-opts "--min-silence-duration=0.03 --min-speech-duration=0.3 --max-speech-duration=10.0" \
  --sad-priors-opts "--sil-scale=0.1" \
  dev_aspire $sad_nnet_dir $sad_nnet_dir \
  data/lang $chain_dir/graph_pp $chain_dir

# Old nnet2-based systems are in the comments below. The results here are
# not applicable to the latest dev set from LDC.

# local/multi_condition/run_nnet2_ms.sh
# 
# local/multi_condition/prep_test_aspire.sh --stage 1 --decode-num-jobs 200 \
#  --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
#  --ivector-scale 0.75 --affix v6 --tune-hyper true dev_aspire data/lang exp/nnet2_multicondition/nnet_ms_a
# 
# local/multi_condition/prep_test_aspire.sh --stage 1 --decode-num-jobs 200 \
#  --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
#  --ivector-scale 0.75 --affix v6 --tune-hyper true test_aspire data/lang exp/nnet2_multicondition/nnet_ms_a
# # 72.3 on leaderboard
# 
# # discriminative training. Helped on dev, but not on dev_test
# local/multi_condition/run_nnet2_ms_disc.sh
# local/multi_condition/prep_test_aspire.sh --stage 1 --decode-num-jobs 200 \
#  --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
#  --ivector-scale 0.75 --affix v6 --tune-hyper true dev_aspire data/lang exp/nnet2_multicondition/nnet_ms_a_smbr_0.00015_nj12
# #%WER 29.1 | 2120 27208 | 77.6 15.4 7.0 6.7 29.1 77.1 | -1.357 | exp/nnet2_multicondition/nnet_ms_c_prior_adjusted_smbr_0.00015_nj12/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_iterepoch2_pp_fg/score_16/penalty_1.0/ctm.filt.filt.sys
# 
# local/multi_condition/prep_test_aspire.sh --stage 1 --decode-num-jobs 200 \
#  --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
#  --ivector-scale 0.75 --affix v6 --tune-hyper true test_aspire data/lang exp/nnet2_multicondition/nnet_ms_a_smbr_0.00015_nj12
# # around 71.5, as models changed after server closed
