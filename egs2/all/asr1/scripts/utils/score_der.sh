#!/bin/bash

# Copyright 2021 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

collar=0.0
frame_shift=128
fs=8000
subsampling=1

. ./utils/parse_options.sh || exit 1

if [ $# -lt 3 ]; then
    echo "Usage: $0 <scoring_dir> <infer_scp> <gt_label>";
    exit 1;
fi

scoring_dir=$1  # scoring directory
infer_scp=$2    # inference scp from diar_inference
gt_label=$3     # ground truth rttm

echo "Scoring at ${scoring_dir}"
mkdir -p $scoring_dir || exit 1;

for med in 1 11; do
    for th in 0.3 0.4 0.5 0.6 0.7; do
        pyscripts/utils/make_rttm.py --median=$med --threshold=$th \
            --frame_shift=${frame_shift} --subsampling=${subsampling} --sampling_rate=${fs} \
            $infer_scp ${scoring_dir}/hyp_${th}_$med.rttm

        md-eval.pl -c ${collar} \
            -r ${gt_label} \
            -s ${scoring_dir}/hyp_${th}_$med.rttm \
            > ${scoring_dir}/result_th${th}_med${med}_collar${collar} 2>/dev/null || exit 1

    done
done

grep OVER $1/result_th0.[^_]*_med[^_]*_collar${collar} \
    | grep -v nooverlap \
    | sort -nrk 7 | tail -n 1
