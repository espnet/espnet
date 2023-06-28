#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000
data_url=www.openslr.org/resources/12
train_set="train_960"
train_dev="dev"
alignment_phoneme_dir="./data/librispeech_phoneme_alignment"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    local/data_asr1.sh --stage 1 --stop-stage 1 --data_url "${data_url}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    local/data_asr1.sh --stage 2 --stop-stage 2
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    local/data_asr1.sh --stage 3 --stop-stage 3 --train_set "${train_set}" --train_dev "${train_dev}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    if [ ! -f ${alignment_phoneme_dir}/dev-clean.tsv ]; then
        log "Stage 4: Downloading  from https://zenodo.org/record/2619474#.Y2F3ZewVDu0"
        mkdir -p ${alignment_phoneme_dir}

        wget \
            -O ${alignment_phoneme_dir}/librispeech_alignments.zip \
            https://zenodo.org/record/2619474/files/librispeech_alignments.zip?download=1

        unzip "${alignment_phoneme_dir}/librispeech_alignments.zip" -d "${alignment_phoneme_dir}"
        python local/dump_librispeech_alignment_from_textgrid.py \
            --alignment_root "${alignment_phoneme_dir}" \
            --dataset "dev-other" "dev-clean"
    else
        log "Stage 4: Librispeech phoneme alignments exists. Skipping ..."
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
