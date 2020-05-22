#!/usr/bin/env bash
#
# Based on the Kaldi LibriCSS and ESPnet LibriSpeech recipes
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0
#
# This script performs recognition with oracle speaker and segment information

# Begin configuration section.
decode_nj=40
stage=0
test_sets=

# ESPnet related variables
dumpdir=dump
do_delta=false
decode_config=conf/decode.yaml
recog_model=model.val5.avg.best
expdir=librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
lang_model=rnnlm.model.best
lmexpdir=exp/train_rnnlm_pytorch_lm_transformer_cosine_batchsize32_lr1e-4_layer16_unigram5000_ngpu4
nbpe=5000
bpemode=unigram
train_set=train_960
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

set -e # exit on error

##########################################################################
# DECODING
##########################################################################

if [ $stage -le 0 ]; then
  pids=() # initialize pids
  for x in $test_sets; do
  (
      x_oracle=${x}_oracle
      decode_dir=decode_${x_oracle}_${recog_model}_$(basename ${decode_config%.*})
      feat_dir=${dumpdir}/${x_oracle}/delta${do_delta}
      # split data
      splitjson.py --parts ${decode_nj} ${feat_dir}/data_${bpemode}${nbpe}.json

      # set batchsize 0 to disable batch decoding
      ${decode_cmd} JOB=1:${decode_nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
          asr_recog.py \
          --config ${decode_config} \
          --ngpu 0 \
          --backend pytorch \
          --batchsize 0 \
          --recog-json ${feat_dir}/split${decode_nj}utt/data_${bpemode}${nbpe}.JOB.json \
          --result-label ${expdir}/${decode_dir}/data.JOB.json \
          --model ${expdir}/results/${recog_model}  \
          --rnnlm ${lmexpdir}/${lang_model} \
          --api v2 \
          --beam-size 30

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
  ) &
  pids+=($!) # store background pids
  done
  i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
  [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
  echo "Decoding done"
fi

##########################################################################
# Scoring: here we obtain wer per condition and overall WER
##########################################################################

if [ $stage -le 1 ]; then
  local/score_reco_oracle.sh \
      --dev ${expdir}/decode_dev_oracle_${recog_model}_$(basename ${decode_config%.*}) \
      --eval ${expdir}/decode_eval_oracle_${recog_model}_$(basename ${decode_config%.*})
fi
