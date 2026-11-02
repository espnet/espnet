#!/usr/bin/env bash
# Reuse the VoxLingua107 LID recipe, then expose GeoLID's directory names.
set -euo pipefail
stage=1
stop_stage=5
. utils/parse_options.sh
bash ../../voxlingua107/lid1/local/data.sh --stage "${stage}" --stop_stage "${stop_stage}"
if [ "${stop_stage}" -ge 5 ]; then
    for split in train dev; do
        source_dir="data/${split}_voxlingua107"
        target_dir="${source_dir}_lang"
        mkdir -p "${target_dir}"
        cp "${source_dir}"/{wav.scp,utt2lang,lang2utt,category2utt} "${target_dir}/"
    done
fi
