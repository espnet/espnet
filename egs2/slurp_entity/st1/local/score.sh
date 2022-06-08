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

SLURP="/ocean/projects/cis210027p/siddhana/slurp"
asr_expdir=$1

if [ $# -gt 1 ]; then
	test_inference_folder=$2
	valid_inference_folder=$3
	python local/convert_to_entity_file.py --exp_root ${asr_expdir} --test_folder ${test_inference_folder} --valid_folder ${valid_inference_folder}
else
	valid_inference_folder=inference_st_model_valid.acc.ave/devel/
	test_inference_folder=inference_st_model_valid.acc.ave/test/
	python local/convert_to_entity_file.py --exp_root ${asr_expdir}
fi
python local/evaluation/evaluate.py -g ${SLURP}/dataset/slurp/test.jsonl -p result_test.json
python local/evaluation/evaluate.py -g ${SLURP}/dataset/slurp/devel.jsonl -p result_valid.json
sclite \
    -r "${asr_expdir}/${valid_inference_folder}/score_bleu/ref.trn.org" trn \
    -h "${asr_expdir}/${valid_inference_folder}/score_bleu/hyp.trn.org" trn \
    -i rm -o all stdout > "${asr_expdir}/${valid_inference_folder}/score_bleu/result_ner.txt"
echo "Write ASR result in ${asr_expdir}/${valid_inference_folder}/score_bleu/result_ner.txt"
grep -e Avg -e SPKR -m 2 "${asr_expdir}/${valid_inference_folder}/score_bleu/result_ner.txt"
sclite \
    -r "${asr_expdir}/${test_inference_folder}/score_bleu/ref.trn.org" trn \
    -h "${asr_expdir}/${test_inference_folder}/score_bleu/hyp.trn.org" trn \
    -i rm -o all stdout > "${asr_expdir}/${test_inference_folder}/score_bleu/result_ner.txt"
echo "Write ASR result in ${asr_expdir}/${test_inference_folder}/score_bleu/result_ner.txt"
grep -e Avg -e SPKR -m 2 "${asr_expdir}/${test_inference_folder}/score_bleu/result_ner.txt"
exit 0

