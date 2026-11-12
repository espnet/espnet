#!/bin/bash

# Copyright 2021 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

collar=0.0
frame_shift=128
fs=8000
subsampling=1
thresholds="0.3 0.4 0.5 0.6 0.7"  # decision thresholds to score
medians="1 11"                     # median filter lengths (frames) to score
uem=        # UEM file of the regions to score (md-eval -u); default: md-eval scores
            # each recording from its first to its last reference segment

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

if [ -n "${uem}" ]; then
    # md-eval cuts a UEM file name at its first ".", so recording ids with a
    # dot (e.g. <session>.CH1) would merge; the UEM, reference and hypotheses
    # given to md-eval have every "." of the recording id replaced by "_".
    # (named by the process id: several scorings of one set may run at once)
    awk '{ gsub(/\./, "_", $1); print }' "${uem}" > "${scoring_dir}/uem.md-eval.$$"
    awk '{ gsub(/\./, "_", $2); print }' "${gt_label}" > "${scoring_dir}/ref.md-eval.$$.rttm"
    gt_label=${scoring_dir}/ref.md-eval.$$.rttm
    uem=${scoring_dir}/uem.md-eval.$$
    trap 'rm -f "${scoring_dir}/uem.md-eval.$$" "${scoring_dir}/ref.md-eval.$$.rttm"' EXIT
fi

for med in ${medians}; do
    for th in ${thresholds}; do
        pyscripts/utils/make_rttm.py --median=$med --threshold=$th \
            --frame_shift=${frame_shift} --subsampling=${subsampling} --sampling_rate=${fs} \
            $infer_scp ${scoring_dir}/hyp_${th}_$med.rttm
        if [ -n "${uem}" ]; then
            awk '{ gsub(/\./, "_", $2); print }' ${scoring_dir}/hyp_${th}_$med.rttm \
                > ${scoring_dir}/hyp_${th}_$med.md-eval.$$.rttm
            mv ${scoring_dir}/hyp_${th}_$med.md-eval.$$.rttm ${scoring_dir}/hyp_${th}_$med.rttm
        fi

        md-eval.pl -c ${collar} ${uem:+-u ${uem}} \
            -r ${gt_label} \
            -s ${scoring_dir}/hyp_${th}_$med.rttm \
            > ${scoring_dir}/result_th${th}_med${med}_collar${collar} 2>/dev/null || exit 1

    done
done

grep OVER $1/result_th0.[^_]*_med[^_]*_collar${collar} \
    | grep -v nooverlap \
    | sort -nrk 7 | tail -n 1
