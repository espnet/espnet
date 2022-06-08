#!/usr/bin/env bash

# Copyright 2022 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
lang=""

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${AMERICASNLP22}" ]; then
    log "Fill the value of 'AMERICASNLP22' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

data_base_url=https://rcweb.dartmouth.edu/homes/f00458c/americasnlp2
lang_name=""

if [ "${lang}" = "bzd" ]; then
    lang_name="Bribri"
elif [ "${lang}" = "gug" ]; then
    lang_name="Guarani"
elif [ "${lang}" = "gvc" ]; then
    lang_name="Kotiria"
elif [ "${lang}" = "qwe" ]; then
    lang_name="Quechua"
elif [ "${lang}" = "tav" ]; then
    lang_name="Waikhana"
else
    log "Language is \"${lang}\", but it must be one of: bzd, gug, gvc, qwe, tav"
    log "Pass a correct language value in the \`--lang\` argument"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage0: Downloading data for AmericasNLP22 (${lang_name})"
    mkdir -p "${AMERICASNLP22}"
    tar_name="${lang_name}TrainDev.tar.gz"
    wget "${data_base_url}/${tar_name}" -O "${AMERICASNLP22}/${tar_name}"
    tar xf "${AMERICASNLP22}/${tar_name}" -C "${AMERICASNLP22}" --overwrite
    rm "${AMERICASNLP22}/${tar_name}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Preparing data for AmericasNLP22 (${lang_name})"

    for dset in train dev; do
        idir="${AMERICASNLP22}/${lang_name}/${dset}"
        odir="data/${dset}_${lang}"
        mkdir -p "${odir}"

        cut -f 1 "${idir}/meta.tsv" \
            | tail -n +2 \
            | sed 's!\(.*\)!\1 '"${idir}/"'\1!' \
            > "${odir}/wav.scp"

        cut -f 1,3 --output-delimiter=" " "${idir}/meta.tsv" \
            | tail -n +2 \
            > "${odir}/text"

        cut -f 1 "${idir}/meta.tsv" \
            | tail -n +2 \
            | sed 's!\(.*\)!\1 \1!' \
            > "${odir}/utt2spk"

        utils/fix_data_dir.sh "${odir}"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
