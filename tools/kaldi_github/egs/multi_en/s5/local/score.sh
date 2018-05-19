#!/bin/bash

###########################################################################################
# This script was copied from egs/fisher_swbd/s5/local/score.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Changed to use steps/score_kaldi.sh instead of local/score_basic.sh
###########################################################################################

# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

orig_args=
for x in "$@"; do orig_args="$orig_args '$x'"; done

# begin configuration section.  we include all the options that score_sclite.sh or
# score_basic.sh might need, or parse_options.sh will die.
# CAUTION: these default values do not have any effect because of the
# way pass things through to the scripts that this script calls.
cmd=run.pl
stage=0
min_lmwt=5
max_lmwt=20
reverse=false
word_ins_penalty=0.0,0.5,1.0
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [options] <data-dir> <lang-dir|graph-dir> <decode-dir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  echo "    --reverse (true/false)          # score with time reversed features "
  exit 1;
fi

data=$1

if [ -f $data/stm ]; then # use sclite scoring.
  echo "$data/stm exists: using local/score_sclite.sh"
  eval local/score_sclite.sh $orig_args
else
  echo "$data/stm does not exist: using steps/score_kaldi.sh"
  eval steps/score_kaldi.sh $orig_args
fi
