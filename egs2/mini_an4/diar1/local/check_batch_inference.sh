#!/usr/bin/env bash
# Check that batched diarization gives each recording what it gets alone.
#
# Runs espnet2.bin.diar_inference on a data directory twice, with batch size 1
# and with a larger batch size, and compares the predictions of every recording.
#
# Usage: local/check_batch_inference.sh [--python python3] [--batch_size 4] \
#            <diar_exp> <inference_config> <data_dir>
set -e
set -u
set -o pipefail

python=python3
batch_size=4

. utils/parse_options.sh

if [ $# -ne 3 ]; then
    echo "Usage: $0 [--python python3] [--batch_size 4] <diar_exp> <inference_config> <data_dir>"
    exit 2
fi
diar_exp=$1
inference_config=$2
data_dir=$3

out=${diar_exp}/check_batch_inference
for bs in 1 "${batch_size}"; do
    ${python} -m espnet2.bin.diar_inference \
        --ngpu 0 \
        --batch_size "${bs}" \
        --config "${inference_config}" \
        --data_path_and_name_and_type "${data_dir}/wav.scp,speech,sound" \
        --train_config "${diar_exp}/config.yaml" \
        --model_file "${diar_exp}/valid.acc.best.pth" \
        --output_dir "${out}/batch${bs}"
done
python3 local/compare_predictions.py "${out}/batch1/diarize.scp" "${out}/batch${batch_size}/diarize.scp"
