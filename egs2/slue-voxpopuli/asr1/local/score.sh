#!/usr/bin/env bash
# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University

# # begin configuration section.
# cmd=run.pl
# stage=0
# data=data/eval2000
# #end configuration section.
score_folder=score_ter

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
else
	valid_inference_folder=$(ls -t ${asr_expdir}/*/devel*/score_wer/hyp.trn | head -n 1 | sed 's!//!/!g' | cut -d/ -f3,4)/
	test_inference_folder=$(ls -t ${asr_expdir}/*/test*/score_wer/hyp.trn | head -n 1 | sed 's!//!/!g' | cut -d/ -f3,4)/
fi

python local/score.py --exp_root ${asr_expdir} \
    --valid_folder ${valid_inference_folder} \
    --test_folder ${test_inference_folder} \
    --score_folder ${score_folder}

sclite \
    -r "${asr_expdir}/${valid_inference_folder}/${score_folder}/ref_asr.trn" trn \
    -h "${asr_expdir}/${valid_inference_folder}/${score_folder}/hyp_asr.trn" trn \
    -i rm -o all stdout > "${asr_expdir}/${valid_inference_folder}/${score_folder}/result_asr.txt"
echo "Write ASR result in ${asr_expdir}/${valid_inference_folder}/${score_folder}/result_asr.txt"
grep -e Avg -e SPKR -m 2 "${asr_expdir}/${valid_inference_folder}/${score_folder}/result_asr.txt"

if [ -d "${test_inference_folder}" ]; then
	sclite \
		-r "${asr_expdir}/${test_inference_folder}/${score_folder}/ref_asr.trn" trn \
		-h "${asr_expdir}/${test_inference_folder}/${score_folder}/hyp_asr.trn" trn \
		-i rm -o all stdout > "${asr_expdir}/${test_inference_folder}/${score_folder}/result_asr.txt"
	echo "Write ASR result in ${asr_expdir}/${test_inference_folder}/${score_folder}/result_asr.txt"
	grep -e Avg -e SPKR -m 2 "${asr_expdir}/${test_inference_folder}/${score_folder}/result_asr.txt"
else
	echo "[Warning] Skip ASR result on test set as it does not exist."
fi

exit 0
