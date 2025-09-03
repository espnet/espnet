#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=5000
data_dir=data
longlibriheavy_repo=https://github.com/Miamoto/LongLibriHeavy.git
longlibriheavy_dir=llh   # local clone dir

log "$0 $*"
. utils/parse_options.sh

# Check required env
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${LIBRILIGHT}" ]; then
    log "Fill the value of 'LIBRILIGHT' in db.sh"
    exit 1
fi

# === Stage 1: Check LIBRILIGHT ===
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -d "${LIBRILIGHT}/small" ] && [ -d "${LIBRILIGHT}/medium" ] && [ -d "${LIBRILIGHT}/large" ]; then
        log "Libri-light found in ${LIBRILIGHT}."
    else
        echo "Some of ${LIBRILIGHT}/{small,medium,large} directories do not exist."
        echo "Please download the LibriLight audio data to ${LIBRILIGHT}:"
        echo "https://github.com/facebookresearch/libri-light/tree/main/data_preparation#1a-downloading"
        exit 1
    fi
fi

# === Stage 2: Clone LongLibriHeavy repo ===
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ -d "${longlibriheavy_dir}/.git" ]; then
        log "LongLibriHeavy already cloned at ${longlibriheavy_dir}, skipping clone."
    else
        log "Cloning LongLibriHeavy benchmark repo..."
        rm -rf ${longlibriheavy_dir}
        git clone ${longlibriheavy_repo} ${longlibriheavy_dir}
        log "Cloned LongLibriHeavy into ${longlibriheavy_dir}"
    fi
fi

# === Stage 3: Copy ESPnet-style data dirs and fix wav.scp paths ===
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Copying LongLibriHeavy data/{train,dev,test} to local data dir and fixing wav.scp paths"
    mkdir -p ${data_dir}

    for split in train dev test; do
        src_dir="${longlibriheavy_dir}/${split}"
        dest_dir="${data_dir}/${split}"

        if [ -d "${src_dir}" ]; then
            cp -r "${src_dir}" "${dest_dir}"
            log "Copied ${split} -> ${dest_dir}"

            # Recursively find and update wav.scp files
            find "${dest_dir}" -type f -name "wav.scp" | while read -r wavscp; do
                sed -i "s|/path/to/audio/root|${LIBRILIGHT}|g" "${wavscp}"
                log "Updated wav.scp: ${wavscp}"
            done
        else
            log "Warning: ${split} directory not found in ${longlibriheavy_dir}"
        fi
    done
fi

# === Stage 4: Check data dir ===
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Rewriting utt2spk with utterance ID as speaker, generating spk2utt, and fixing data dirs"

    for split in train dev test; do
        split_dir="${data_dir}/${split}"

        if [ -d "${split_dir}" ]; then
            find "${split_dir}" -type f -name "utt2spk" | while read -r utt2spk_path; do
                dir_path=$(dirname "${utt2spk_path}")
                spk2utt_path="${dir_path}/spk2utt"

                # Rewrite utt2spk: duplicate utterance ID as speaker ID, then sort
                awk '{ print $1, $1 }' "${utt2spk_path}" | sort > "${utt2spk_path}.tmp" || {
                    log "Error: Failed to create temporary utt2spk file at ${utt2spk_path}.tmp"
                    exit 1
                }
                mv "${utt2spk_path}.tmp" "${utt2spk_path}" || {
                    log "Error: Failed to move temporary utt2spk file to ${utt2spk_path}"
                    exit 1
                }
                log "Rewritten and sorted utt2spk in ${utt2spk_path} with utterance ID as speaker"

                # Generate spk2utt and sort it
                utils/utt2spk_to_spk2utt.pl "${utt2spk_path}" | sort > "${spk2utt_path}"
                log "Generated and sorted spk2utt at ${spk2utt_path}"

                # Fix data directory to ensure consistency
                utils/fix_data_dir.sh "${dir_path}" || {
                    log "Error fixing data dir: ${dir_path}"
                    exit 1
                }
                log "Fixed data dir: ${dir_path}"
            done
        else
            log "Warning: Split directory not found: ${split_dir}"
        fi
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
