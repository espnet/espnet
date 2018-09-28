#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

backend=pytorch
do_delta=false 
verbose=0 
debugmode=1


# decoding parameter
beam_size=20
penalty=0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'
nj=32
lm=
lm_weight=0.0

. ./utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: ./local/decode.sh <exp> <dict> <train_cmvn> <data>"
  exit 1
fi

expdir=$1
dict=$2
train_cmvn=$3
data=$4 #data/set1 data/set2

echo "stage 4: Decoding"

extra_opts=
if [ ! -z $lm ]; then
  extra_opts="--rnnlm ${lm}/rnnlm.model.best --lm-weight ${lm_weight}"
fi

for d in $data; do
(
  rtask=`basename ${d}`
  decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
  if [ ! -z $lm ]; then
    decode_dir="${decode_dir}_rnnlm${lm_weight}"
  fi
  # split data
  split_data.sh --per-utt ${d} ${nj};
  sdata=${d}/split${nj}utt;

  # feature extraction
  feats="ark,s,cs:apply-cmvn --norm-vars=true ${train_cmvn} scp:${sdata}/JOB/feats.scp ark:- |"
  if ${do_delta}; then
    feats="$feats add-deltas ark:- ark:- |"
  fi

  # make json labels for recognition
  #data2json.sh ${d} ${dict} > ${d}/data.json

  ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
      asr_recog.py \
      --ngpu 0 \
      --backend ${backend} \
      --debugmode ${debugmode} \
      --verbose ${verbose} \
      --recog-feat "$feats" \
      --recog-label ${d}/data.json \
      --result-label ${expdir}/${decode_dir}/data.JOB.json \
      --model ${expdir}/results/model.${recog_model}  \
      --model-conf ${expdir}/results/model.conf  \
      --beam-size ${beam_size} \
      --penalty ${penalty} \
      --maxlenratio ${maxlenratio} \
      --minlenratio ${minlenratio} \
      --ctc-weight ${ctc_weight} \
      ${extra_opts} &
  wait

  score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
) &
done
wait
echo "Finished"

