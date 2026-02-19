#!/usr/bin/env bash

# Copyright 2024 CMU WAVLab (Srishti Ginjala)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Data preparation script for CommonVoice gender-based subsets
# This script filters CommonVoice English data by gender attribute
# to create demographic-specific ASR training/evaluation sets.

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# General configuration
stage=0
stop_stage=100
SECONDS=0
lang=en
gender=male  # "male" or "female" -- the gender subset to create

. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${COMMONVOICE}" ]; then
    log "Fill the value of 'COMMONVOICE' of db.sh"
    exit 1
fi
mkdir -p ${COMMONVOICE}

set -e
set -u
set -o pipefail

# Determine the CommonVoice corpus version directory
# You may need to adjust this path based on your downloaded version
cv_dir="${COMMONVOICE}"
# Try to find the actual corpus directory (handles different CV versions)
if [ -d "${cv_dir}/cv-corpus-24.0-2025-12-05/en" ]; then
    cv_data_dir="${cv_dir}/cv-corpus-24.0-2025-12-05/en"
elif [ -d "${cv_dir}/cv-corpus-17.0-2024-03-15/en" ]; then
    cv_data_dir="${cv_dir}/cv-corpus-17.0-2024-03-15/en"
elif [ -d "${cv_dir}/en" ]; then
    cv_data_dir="${cv_dir}/en"
else
    # Try to find any cv-corpus directory
    cv_data_dir=$(find "${cv_dir}" -maxdepth 2 -name "validated.tsv" -exec dirname {} \; | head -1)
    if [ -z "${cv_data_dir}" ]; then
        log "Error: Cannot find CommonVoice data in ${cv_dir}"
        log "Expected structure: ${cv_dir}/<cv-corpus-version>/${lang}/"
        log "  with files: validated.tsv, test.tsv, dev.tsv, clips/"
        exit 1
    fi
fi

log "Using CommonVoice data from: ${cv_data_dir}"

train_set="train_${gender}_${lang}"
train_dev="dev_${gender}_${lang}"
test_set="test_${gender}_${lang}"

log "Data preparation started for gender=${gender}, lang=${lang}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Verifying CommonVoice data exists"
    for f in validated.tsv test.tsv dev.tsv; do
        if [ ! -f "${cv_data_dir}/${f}" ]; then
            log "Error: ${cv_data_dir}/${f} not found"
            log "Please download CommonVoice English data from https://commonvoice.mozilla.org/"
            exit 1
        fi
    done
    log "Data files verified."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Preparing gender-filtered data for CommonVoice"

    # Create gender-filtered data directories using Python script
    for part in "validated" "test" "dev"; do
        out_dir="data/$(echo "${part}_${gender}_${lang}" | tr - _)"
        log "Processing ${part} -> ${out_dir} (filtering for gender=${gender})"

        python3 local/data_prep_gender.py \
            --cv_dir "${cv_data_dir}" \
            --split "${part}" \
            --gender "${gender}" \
            --out_dir "${out_dir}"
    done

    # Remove test & dev data from the validated (training) set
    utils/copy_data_dir.sh "data/$(echo "validated_${gender}_${lang}" | tr - _)" "data/${train_set}"
    utils/filter_scp.pl --exclude "data/${train_dev}/wav.scp" \
        "data/${train_set}/wav.scp" > "data/${train_set}/temp_wav.scp"
    utils/filter_scp.pl --exclude "data/${test_set}/wav.scp" \
        "data/${train_set}/temp_wav.scp" > "data/${train_set}/wav.scp"
    rm -f "data/${train_set}/temp_wav.scp"
    utils/fix_data_dir.sh "data/${train_set}"

    # Subsample training set to ~100 hours
    # CommonVoice clips average ~5 seconds, so 100 hours ≈ 72,000 utterances
    max_utts=72000
    n_train=$(wc -l < "data/${train_set}/wav.scp")
    if [ ${n_train} -gt ${max_utts} ]; then
        log "Subsampling training set from ${n_train} to ${max_utts} utterances (~100 hours)"
        utils/subset_data_dir.sh "data/${train_set}" ${max_utts} "data/${train_set}_full"
        # swap: move subset into main train dir
        mv "data/${train_set}" "data/${train_set}_all"
        mv "data/${train_set}_full" "data/${train_set}"
    fi

    # Print statistics
    for d in "data/${train_set}" "data/${train_dev}" "data/${test_set}"; do
        if [ -d "${d}" ]; then
            n_utts=$(wc -l < "${d}/wav.scp")
            log "  ${d}: ${n_utts} utterances"
        fi
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
