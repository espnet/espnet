#!/usr/bin/env bash

# Copyright 2022 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir> <valid_inference_folder> <test_inference_folder>"
  exit 1;
fi

asr_expdir=$1
if [ $# -gt 1 ]; then
        valid_inference_folder=$2
        test_inference_folder=$3
else
        valid_inference_folder="decode_asr_asr_model_valid.loss.ave/valid_context3/"
        test_inference_folder="decode_asr_asr_model_valid.loss.ave/test_context3/"
fi

python local/score.py \
		--exp_root ${asr_expdir} \
		--valid_folder ${valid_inference_folder} \
		--test_folder ${test_inference_folder} \
	| sed 's/Intent/Dialog Act/g'

exit 0

