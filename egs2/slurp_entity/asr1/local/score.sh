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
  echo "Usage: local/score.sh <asr-exp-dir> <valid_inference_folder> <test_inference_folder>"
  exit 1;
fi
. ./db.sh

asr_expdir=$1

if [ $# -gt 1 ]; then
	valid_inference_folder=$2
	test_inference_folder=$3
	python local/score.py --exp_root ${asr_expdir} --valid_folder ${valid_inference_folder} --test_folder ${test_inference_folder}
	python local/convert_to_entity_file.py --exp_root ${asr_expdir} --valid_folder ${valid_inference_folder} --test_folder ${test_inference_folder}
else
	python local/score.py --exp_root ${asr_expdir}
	python local/convert_to_entity_file.py --exp_root ${asr_expdir}
fi
python local/evaluation/evaluate.py -g ${SLURP}/dataset/slurp/test.jsonl -p ${asr_expdir}/result_test.json
exit 0
