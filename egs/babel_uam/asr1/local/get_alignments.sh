#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

###############################################################################
# This script trains a HMM-GMM system for the purpose of obtaining phonetic 
# transcripts. The goal of training to predict phoneme strings is for use in
# zero-shot ASR based on cross lingual transfer of phonemes. It is almost
# exactly the HMM-GMM training through tri5 of the primary babel recipe applied
# to all the data across all languages.
###############################################################################

set -e
set -o pipefail
. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

if [ ! -f exp/mono/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) monophone training in exp/mono on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mono.sh \
    --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
    data/train_sub1 data/lang_universal exp/mono
  touch exp/mono/.done
fi


if [ ! -f exp/tri1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) triphone training in exp/tri1 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
    data/train_sub2 data/lang_universal exp/mono exp/mono_ali_sub2

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
    data/train_sub2 data/lang_universal exp/mono_ali_sub2 exp/tri1

  touch exp/tri1/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (medium) triphone training in exp/tri2 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri2/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
    data/train_sub3 data/lang_universal exp/tri1 exp/tri1_ali_sub3

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
    data/train_sub3 data/lang_universal exp/tri1_ali_sub3 exp/tri2

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
    data/train_sub3 data/lang_universal data/dict_universal \
    exp/tri2 data/dict_universal/dictp/tri2 data/dict_universal/langp/tri2 data/lang_universalp/tri2

  touch exp/tri2/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (full) triphone training in exp/tri3 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri3/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang_universalp/tri2 exp/tri2 exp/tri2_ali

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesTri3 $numGaussTri3 data/train data/lang_universalp/tri2 exp/tri2_ali exp/tri3

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
    data/train data/lang_universal data/dict_universal/ \
    exp/tri3 data/dict_universal/dictp/tri3 data/dict_universal/langp/tri3 data/lang_universalp/tri3

  touch exp/tri3/.done
fi


echo ---------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri4/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang_universalp/tri3 exp/tri3 exp/tri3_ali

  steps/train_lda_mllt.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train data/lang_universalp/tri3 exp/tri3_ali exp/tri4

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
    data/train data/lang_universal data/dict_universal \
    exp/tri4 data/dict_universal/dictp/tri4 data/dict_universal/langp/tri4 data/lang_universalp/tri4

  touch exp/tri4/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (SAT) triphone training in exp/tri5 on" `date`
echo ---------------------------------------------------------------------

if [ ! -f exp/tri5/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang_universalp/tri4 exp/tri4 exp/tri4_ali

  steps/train_sat.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train data/lang_universalp/tri4 exp/tri4_ali exp/tri5

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
    data/train data/lang_universal data/dict_universal \
    exp/tri5 data/dict_universal/dictp/tri5 data/dict_universal/langp/tri5 data/lang_universalp/tri5

  touch exp/tri5/.done
fi

if [ ! -f exp/tri5_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/tri5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang_universalp/tri5 exp/tri5 exp/tri5_ali

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
    data/train data/lang_universal data/dict_universal \
    exp/tri5_ali data/dict_universal/dictp/tri5_ali data/dict_universal/langp/tri5_ali data/lang_universalp/tri5_ali

  touch exp/tri5_ali/.done
fi


ali-to-phones exp/tri5_ali/final.mdl ark:"gunzip -c exp/tri5_ali/ali.*.gz |" ark,t:- |\
  sort | ./utils/int2sym.pl -f 2- data/lang_universalp/tri5_ali/phones.txt |\
  sed 's/_[BISE] / /g' > data/data.ali.phn

exit 0;
