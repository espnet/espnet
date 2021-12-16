#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1
stop_stage=100
verbose=1
n_gpus=1
n_jobs=16

# config file
conf=

# directory path setting
download_dir=../vc1_task1/downloads  # set this depending on which task you participate in
list_dir=../vc1_task1/local/lists    # set this depending on which task you participate in
dumpdir=dump                         # directory to dump features

# training related setting
tag=""      # tag to manage expreiments
resume=""   # checkpoint file to resume to
task=       # "task1" or "task2"

# decoding related setting
checkpoint=""

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train_${task}"
dev_set="dev_${task}"

set -euo pipefail

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo "Stage -1: Data download"
    echo "Please download the dataset in either vc1_task1 or vc1_task2."
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"

    if [ ! -e ${download_dir} ]; then
        echo "${download_dir} not found."
        echo "cd ${download_dir}; ./run.sh --stop_stage -1; cd -"
        exit 1;
    fi

    local/data_prep.sh \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        "${download_dir}" "${list_dir}" data ${task}
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    # extract raw features
    pids=()
    for name in "${train_set}" "${dev_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        local/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            parallel-wavegan-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

    # calculate statistics for normalization
    echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
    ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
        parallel-wavegan-compute-statistics \
            --config "${conf}" \
            --rootdir "${dumpdir}/${train_set}/raw" \
            --dumpdir "${dumpdir}/${train_set}" \
            --verbose "${verbose}"
    echo "Successfully finished calculation of statistics."

    # normalize and dump them
    pids=()
    for name in "${train_set}" "${dev_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm" ] && mkdir -p "${dumpdir}/${name}/norm"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm/normalize.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm/normalize.JOB.log" \
            parallel-wavegan-normalize \
                --config "${conf}" \
                --stats "${dumpdir}/${train_set}/stats.h5" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished normalization."
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_${tag}"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "${dumpdir}/${train_set}/stats.h5" "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python3 -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/norm" \
            --dev-dumpdir "${dumpdir}/${dev_set}/norm" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    [ ! -e "${outdir}/${dev_set}" ] && mkdir -p "${outdir}/${name}"
    [ "${n_gpus}" -gt 1 ] && n_gpus=1
    echo "Decoding start. See the progress via ${outdir}/${dev_set}/decode.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${dev_set}/decode.log" \
        parallel-wavegan-decode \
            --dumpdir "${dumpdir}/${dev_set}/norm" \
            --checkpoint "${checkpoint}" \
            --outdir "${outdir}/${dev_set}" \
            --verbose "${verbose}"
    echo "Successfully finished decoding of ${dev_set} set."
    echo "Successfully finished decoding."
fi
