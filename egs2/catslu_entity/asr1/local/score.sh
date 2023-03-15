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
. ./db.sh

asr_expdir=$1

decode_folder=$(dirname $(dirname $(ls -t ${asr_expdir}/*/test_*/score_wer  | head -n1 )))

python local/score.py ${decode_folder} > ${decode_folder}/score_slu.txt

cat ${decode_folder}/score_slu.txt

exit 0
