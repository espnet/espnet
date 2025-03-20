#!/usr/bin/env bash
# Copyright 2025  Siddhant Arora
#           2025  Carnegie Mellon University
# Apache 2.0

# begin configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir>"
  exit 1;
fi

asr_expdir=$1
python local/score_turn_take.py --exp_root ${asr_expdir}

exit 0
