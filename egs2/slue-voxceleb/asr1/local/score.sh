#!/usr/bin/env bash
# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University

# # begin configuration section.
# cmd=run.pl
# stage=0
# data=data/eval2000
# #end configuration section.

# TODO(siddhana): Automatically determine the decoding folder name
# TODO(siddhana): Show SLU results in RESULTS.md

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
	python local/score.py --exp_root ${asr_expdir} --valid_folder ${valid_inference_folder} --test_folder ${test_inference_folder}
	python local/generate_asr_files.py --exp_root ${asr_expdir} --valid_folder ${valid_inference_folder} --test_folder ${test_inference_folder}
	python local/f1_score.py --exp_root ${asr_expdir} --valid_folder ${valid_inference_folder} --test_folder ${test_inference_folder}
else
	valid_inference_folder="inference_asr_model_valid.acc.ave_10best/devel/"
	test_inference_folder="inference_asr_model_valid.acc.ave_10best/test/"
	python local/score.py --exp_root ${asr_expdir}
	python local/generate_asr_files.py --exp_root ${asr_expdir}
	python local/f1_score.py --exp_root ${asr_expdir}
fi

sclite \
            -r "${asr_expdir}/${valid_inference_folder}/score_wer/ref_asr.trn" trn \
            -h "${asr_expdir}/${valid_inference_folder}/score_wer/hyp_asr.trn" trn \
            -i rm -o all stdout > "${asr_expdir}/${valid_inference_folder}/score_wer/result_asr.txt"
echo "Write ASR result in ${asr_expdir}/${valid_inference_folder}/score_wer/result_asr.txt"
                grep -e Avg -e SPKR -m 2 "${asr_expdir}/${valid_inference_folder}/score_wer/result_asr.txt"

sclite \
            -r "${asr_expdir}/${test_inference_folder}/score_wer/ref_asr.trn" trn \
            -h "${asr_expdir}/${test_inference_folder}/score_wer/hyp_asr.trn" trn \
            -i rm -o all stdout > "${asr_expdir}/${test_inference_folder}/score_wer/result_asr.txt"
echo "Write ASR result in ${asr_expdir}/${test_inference_folder}/score_wer/result_asr.txt"
                grep -e Avg -e SPKR -m 2 "${asr_expdir}/${test_inference_folder}/score_wer/result_asr.txt"

exit 0
