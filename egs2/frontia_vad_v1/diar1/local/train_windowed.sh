#!/usr/bin/env bash
# Train on fixed-length windows instead of whole recordings (run.sh stage 5).
#
# Builds dump/raw/<train_set>_seg<window> and dump/raw/<valid_set>_seg160000
# (local/make_windows.py: wav.scp + Kaldi segments + vad_intervals) from the sets
# that diar.sh dumped, then runs diar_train on the windows' audio
# (omniio_segments) and labels (vad_intervals), with the global MVN stats of
# diar.sh stage 4. The model goes to exp/diar_<diar_tag>, where diar.sh stages
# 6-7 (--diar_tag) find it. Arguments after the options go to diar_train.
set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

config=conf/train.yaml
diar_tag=train
train_set=train
valid_set=dev
window=160000   # samples: 10 s at 16 kHz
ngpu=1
fs=16k

while [ $# -gt 0 ]; do
    case "$1" in
        --config) config=$2; shift 2 ;;
        --diar_tag) diar_tag=$2; shift 2 ;;
        --train_set) train_set=$2; shift 2 ;;
        --valid_set) valid_set=$2; shift 2 ;;
        --window) window=$2; shift 2 ;;
        --ngpu) ngpu=$2; shift 2 ;;
        --fs) fs=$2; shift 2 ;;
        *) break ;;
    esac
done

. ./path.sh
. ./cmd.sh

stats=exp/diar_stats_${fs}/train/feats_stats.npz
[ -f "${stats}" ] || { log "No ${stats}: run diar.sh up to stage 4 first"; exit 1; }

# training windows of ${window} samples; validation always on 10 s windows so
# that runs with different window lengths are compared on the same set
train_dir=dump/raw/${train_set}_seg${window}
valid_dir=dump/raw/${valid_set}_seg160000
[ -s "${train_dir}/segments" ] || python3 local/make_windows.py "dump/raw/${train_set}" \
    "${train_dir}" --window "${window}" --shift "${window}"
[ -s "${valid_dir}/segments" ] || python3 local/make_windows.py "dump/raw/${valid_set}" \
    "${valid_dir}" --window 160000 --shift 160000

output_dir=exp/diar_${diar_tag}
mkdir -p "${output_dir}"
log "Training ${output_dir} (log: ${output_dir}/train.log)"
# shellcheck disable=SC2154
${cuda_cmd} --gpu "${ngpu}" "${output_dir}/train.log" \
    python3 -m espnet2.bin.diar_train \
    --use_preprocessor true \
    --resume true \
    --output_dir "${output_dir}" \
    --config "${config}" \
    --frontend_conf fs="${fs}" \
    --normalize global_mvn --normalize_conf "stats_file=${stats}" \
    --num_spk 1 \
    --train_data_path_and_name_and_type "${train_dir}/segments,speech,omniio_segments" \
    --train_data_path_and_name_and_type "${train_dir}/vad_intervals,spk_labels,vad_intervals" \
    --train_shape_file "${train_dir}/speech_shape" \
    --valid_data_path_and_name_and_type "${valid_dir}/segments,speech,omniio_segments" \
    --valid_data_path_and_name_and_type "${valid_dir}/vad_intervals,spk_labels,vad_intervals" \
    --valid_shape_file "${valid_dir}/speech_shape" \
    --ngpu "${ngpu}" \
    "$@"
