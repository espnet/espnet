#!/usr/bin/env bash
# Copyright 2021 Carnegie Mellon University (Author : Roshan Sharma)

## begin configuration section.
stage=0
data=data/eval2000
 #end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir> <valid_inference_folder> <test_inference_folder>"
  exit 1;
fi

asr_expdir=$1


name=$(basename ${data}) # e.g. eval2000
for dir in ${asr_expdir}/decode_*/${name}/score_wer; do
    python ../../../utils/score_summarization.py $data/sum $dir/text
done   