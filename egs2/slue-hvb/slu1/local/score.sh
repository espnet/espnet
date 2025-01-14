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

if [ $# -gt 3 ]; then
  echo "Run only classifier scoring"
  use_only_classifier=$1
  asr_expdir=$2
  valid_inference_folder=$3
  test_inference_folder=$4
  if ${use_only_classifier}; then
    echo "Run only classifier scoring"
    python local/score.py --exp_root ${asr_expdir} --valid_folder ${valid_inference_folder} --test_folder ${test_inference_folder} --use_only_classifier ${use_only_classifier}
    exit 0
  fi
else
    asr_expdir=$1
    valid_inference_folder=$2
	  test_inference_folder=$3
fi

python local/score.py --exp_root ${asr_expdir} --valid_folder ${valid_inference_folder} --test_folder ${test_inference_folder}
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
