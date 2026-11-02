#!/usr/bin/env bash
# Prepare and dump each corpus before filtering Babel and combining training data.
set -euo pipefail

stage=1
stop_stage=3
nj=8
python=python3
local_data_opts=

help_message="Usage: $0 [--stage N] [--stop_stage N] [--nj N]
Options: --python, --local_data_opts.
Runs shared LID stages 1-3, then filters Babel and combines the training dumps."
. utils/parse_options.sh
if [ $# -ne 0 ]; then
    echo "${help_message}" >&2
    exit 2
fi
if ! [[ "${stage}" =~ ^[0-9]+$ && "${stop_stage}" =~ ^[0-9]+$ ]]; then
    echo "$0: stage and stop_stage must be non-negative integers" >&2
    exit 2
fi

if [ "${stop_stage}" -gt 3 ]; then
    stop_stage=3
fi
if [ "${stage}" -gt "${stop_stage}" ]; then
    exit 0
fi

for dataset in voxlingua107 fleurs ml_superb2 babel voxpopuli; do
    case "${dataset}" in
        fleurs|voxpopuli) test_sets="test_${dataset}_lang" ;;
        ml_superb2) test_sets=dev_dialect_ml_superb2_lang ;;
        *) test_sets= ;;
    esac
    ./lid.sh \
        --stage "${stage}" --stop_stage "${stop_stage}" \
        --nj "${nj}" --python "${python}" \
        --feats_type raw --fs 16k \
        --expdir exp_combined \
        --train_set "train_${dataset}_lang" \
        --valid_set "dev_${dataset}_lang" \
        --test_sets "${test_sets}" \
        --local_data_opts "${local_data_opts} --datasets ${dataset}"
done

if [ "${stop_stage}" -ge 3 ]; then
    bash local/filter_babel_over_10s.sh --python "${python}"
    bash local/combine.sh --python "${python}"
fi
