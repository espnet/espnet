#!/usr/bin/env bash
# Copyright 2022  University of Stuttgart (Pavel Denisov)

# # begin configuration section.
# cmd=run.pl
# stage=0
# data=data/eval2000
# #end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir>"
  exit 1;
fi

asr_expdir=$1

for dset in "dev" "test"; do
  score_folder=$(dirname $(ls -t ${asr_expdir}/*/${dset}/score_wer/ref.trn  | head -n1))
  python local/score.py ${score_folder}/ref.trn ${score_folder}/hyp.trn > $(dirname ${score_folder})/score_slu.txt
done

grep CVER ${asr_expdir}/*/*/score_slu.txt

exit 0
