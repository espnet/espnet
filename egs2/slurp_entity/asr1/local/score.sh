#!/usr/bin/env bash
# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University

# # begin configuration section.
# cmd=run.pl
# stage=0
# data=data/eval2000
# #end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir> <inference_config>"
  exit 1;
fi

asr_expdir=$1

if [ $# -gt 1 ]; then
	inference_config=$2
	python local/score.py ${asr_expdir} ${inference_config}
else
	python local/score.py ${asr_expdir}
fi

exit 0

