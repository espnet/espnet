#!/usr/bin/env bash
# Copyright 2021 Carnegie Mellon University (Yifan Peng)

asr_tag=
inference_tag=
test_sets=

. utils/parse_options.sh
. ./path.sh

inference_expdir=exp/asr_${asr_tag}/${inference_tag}
acc_file=${inference_expdir}/accuracy.csv
echo "name,total,correct,accuracy" | tee ${acc_file}
for x in ${test_sets}; do
    wer_dir=${inference_expdir}/${x}/score_wer
    python local/score.py --wer_dir ${wer_dir}
    echo "${x},$(tail -n 1 ${wer_dir}/../accuracy.csv)" | tee -a ${acc_file} || exit 1
done

echo "$0: Successfully wrote accuracy results to file ${acc_file}"
