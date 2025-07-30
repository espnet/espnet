#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# This script expects the DYNAMIC_SUPERB environment variable to be set.
# You can set it in db.sh or export it in your shell.
if [ -z "${DYNAMIC_SUPERB}" ]; then
    echo "Error: DYNAMIC_SUPERB environment variable is not set." >&2
    exit 1
fi

core_tasks=("SuperbASR_LibriSpeech-TestClean")


for task in "${core_tasks[@]}"; do
    data_dir=data/${task}
    mkdir -p ${data_dir}

    echo "Processing task: ${task}"
    meta_file="$DYNAMIC_SUPERB/${task}.json"
    python pyscripts/utils/download_dynamic_superb.py \
        --json_path "${meta_file}" \
        --save_dir "${data_dir}" \
        --allow_multi_utt

    # Create spk2utt from utt2spk
    utils/utt2spk_to_spk2utt.pl "${data_dir}/utt2spk" > "${data_dir}/spk2utt"

done

echo "Data preparation finished."
