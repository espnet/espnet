#!/usr/bin/env bash
# Copyright 2021 Carnegie Mellon University (Author : Roshan Sharma)

## begin configuration section.
data=data/dev5_test_sum
# end configuration section.


[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir>"
  exit 1;
fi


asr_expdir=$1

name=$(basename ${data}) # e.g. dev5_test
echo "${asr_expdir}/decode_*/${name}"
for dir in ${asr_expdir}/decode_*/${name}; do
    python pyscripts/utils/score_summarization.py $data/text $dir/text $(echo $dir | sed 's/exp//g') > $dir/result.sum
done   
