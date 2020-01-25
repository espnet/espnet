#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
SECONDS=0

stage=-1
stop_stage=1

# it can be augmented depending on avalaible GPU memory
maxframes=3000
maxchars=400

. ./db.sh
. ./path.sh
. ./cmd.sh

data_how2_text=${HOW2_TEXT}
data_how2_feats=${HOW2_FEATS}
data_iwslt19=${HOW2_TEXT}/test_set_iwslt2019

# url to download iwslt 2019 test set
url_iwslt19=http://islpc21.is.cs.cmu.edu/ramons/iwslt2019.tar.gz

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ! -d ${data_how2_text} ]; then
    log "${data_how2_text} doesn't exist. Please read instructions in README.md."
    exit 2
fi

if [[ $(find ${data_how2_text}/features -type f -name '*.ark' | wc -l) -gt 0 ]]; then
    log "It seems ${data_how2_text}/features already contains ark files. Skipping copy."
else
    if [ ! -d ${data_how2_feats} ]; then
        log "${data_how2_feats} doesn't exist. Please read instructions in README.md."
        exit 2
    else
        cp ${data_how2_feats}/* ${data_how2_text}/features/
    fi
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data download"

    curr_path=${PWD}

    if [ -d ${data_iwslt19} ] || [ -f ${data_how2_text}/iwslt2019.tar.gz ]; then
        log "$0: iwslt2019 directory or archive already exists in ${data_how2_text}. Skipping download."
    else
        if ! which wget >/dev/null; then
            log "$0: wget is not installed."
            exit 2
        fi
        log "$0: downloading test set from ${url_iwslt19}"

        cd ${data_how2_text}
        if ! wget --no-check-certificate ${url_iwslt19}; then
            log "$0: error executing wget ${url_iwslt19}"
            exit 2
        fi

        if ! tar -xvzf iwslt2019.tar.gz; then
            log "$0: error un-tarring archive ${data_how2_text}/iwslt19.tar.gz"
            exit 2
        fi
        cd ${curr_path}

        log "$0: Successfully downloaded and un-tarred ${data_how2_text}/iwslt19.tar.gz"
    fi
fi

match_files () {
    workdir=${1}
    file=${2}
    dstdir=${3}

    mv ${dstdir}/${file} ${dstdir}/${file}.old

    awk 'NR==FNR{l[$0];next;} !(FNR in l)' ${workdir}/utterances.deleted \
        ${dstdir}/${file}.old > ${dstdir}/${file}

    if [ $file == "utt2spk" ]; then
        utils/utt2spk_to_spk2utt.pl ${dstdir}/${file} > ${dstdir}/spk2utt
    fi
}

check_files () {
    # dummy version of utils/validate_data_dir.sh except we use text for id-comparison
    dir=${1}
    text=${2}
    files="utt2spk segments feats.scp"

    [[ ! -f ${dir}/${text} ]] && log "$0: ${dir}/${text} should exist" && exit 2
    n_utts=$(wc -l < ${dir}/${text})

    for x in ${files}; do
        [[ ! -f ${dir}/${x} ]] && log "$0: ${dir}/${x} should exist" && exit 2

        seen_entries=$(awk '!seen[$1]++' ${dir}/${text} ${dir}/${x} | wc -l)

        [ "${seen_entries}" -ne "${n_utts}" ] &&
            log "Error: ID mismatch between ${dir}/${text} and ${dir}/${x}" && exit 2
    done

    return 0
}

prepare_set () {
    # main function: punctuation, text and case normalizations are applied
    srcdir=${1}
    dstdir=${2}
    workdir=${2}/normalize
    text=${3}
    pattern=${4}
    new_path=${5}

    [ ! -d ${srcdir} ] && log "$0: No such directory ${srcdir}" && exit 2
    [ -d ${dstdir} ] && rm -rf ${2}
    mkdir -p ${workdir}

    check_files ${srcdir} ${text} && cp -r ${srcdir}/* ${dstdir}

    # re-write paths in feats.scp and wav.scp with user paths
    for x in feats.scp cmvn.scp; do
        if [[ ${srcdir} == /* ]]; then
            sed -i "s|${pattern}|${new_path}|g" ${dstdir}/${x}
        else
            sed -i "s|${pattern}|${PWD}/${new_path}|g" ${dstdir}/${x}
        fi
    done

    cut -d' ' -f1 ${dstdir}/${text} > ${workdir}/id.en
    cut -d' ' -f2- ${dstdir}/${text} > ${workdir}/text.en

    # normalize punctuation using moses decoder
    normalize-punctuation.perl -l en < ${workdir}/text.en > ${workdir}/text.en.norm

    # normalize text and modify other needed files (utt2spk, spk2utt..) according to new text file
    # excluding punctiation normalization iwslt19 test set remains unchanged.
    if [[ ${dstdir} == *"test_set_iwslt2019"* ]]; then
        cp ${workdir}/id.en ${workdir}/id.en.norm
        cp ${workdir}/text.en.norm ${workdir}/text.en.norm2
    else
        local/normalize_how2_texts.sh ${workdir}/text.en.norm ${workdir}/id.en \
                                      ${workdir}/text.en.norm2 ${workdir}/id.en.norm

        # utterances removed during text normalization are also removed in other files
        for x in segments utt2spk feats.scp; do
            match_files ${workdir} ${x} ${dstdir}
        done
    fi

    # saving lowercase and uppercase version of text. uppercase is for iwslt19
    tr '[:upper:]' '[:lower:]' < ${workdir}/text.en.norm2 > ${workdir}/text.en.norm2.lc
    tr '[:lower:]' '[:upper:]' < ${workdir}/text.en.norm2 > ${workdir}/text.en.norm2.uc

    paste -d' ' ${workdir}/id.en.norm ${workdir}/text.en.norm2.lc > ${dstdir}/text
    paste -d' ' ${workdir}/id.en.norm ${workdir}/text.en.norm2.uc > ${dstdir}/text.uc

    check_files ${dstdir} text
}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation (HOW2 + IWSLT 2019 test set)"

    feats_path=${data_how2_text}/features
    feats_pattern="ARK_PATH"

    for set in train val dev5 test_set_iwslt2019; do
        srcdir=${data_how2_text}/data/${set}
        dstdir=data/${set}
        text="text.id.en"

        if [[ $set == "test_set_iwslt2019" ]]; then
            feats_path=${data_iwslt19}
            feats_pattern="\.\."
            srcdir=${data_iwslt19}/test
            text="text"
        fi

        prepare_set ${srcdir} ${dstdir} ${text} ${feats_pattern} ${feats_path}

        if [ $set == "train" ]; then
            remove_longshortdata.sh --maxframes ${maxframes}        \
                                    --maxchars ${maxchars}          \
                                    data/${set}                     \
                                    data/${set}_reduced
        fi
    done

    # add non-linguistic symbols
    echo "[hes]" > data/nlsyms
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
