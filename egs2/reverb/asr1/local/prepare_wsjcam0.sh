#!/usr/bin/env bash
set -euo pipefail
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
help_message="Usage $0 <wsjcam0>"

# To specify the sort order
export LC_ALL=C

log "$0 $*"
. utils/parse_options.sh || exit 1;


if [ $# -ne 1 ]; then
    log "Error: invalid command line arguments"
    log "${help_message}"
    exit 1
fi

. ./path.sh  # Setup the environment

WSJCAM0=$1
train_set_org=si_tr
eval_sets_org="si_dt si_et_1 si_et_2"

# 1. text
for dset in ${train_set_org} ${eval_sets_org}; do
    mkdir -p data/wsjcam0_${dset}
    if [ "${dset}" == si_et_2 ];then
        mic=secondary_microphone
    else
        mic=primary_microphone
    fi

    cat ${WSJCAM0}/data/${mic}/${dset}*/*/*.dot \
        | local/dot2scp.py | local/normalize_transcript.pl "<NOISE>" > data/wsjcam0_${dset}/text.org

    # Remove not speaking utterances (I don't know why it exists.)
    <data/wsjcam0_${dset}/text.org \
        awk '{ if ( NF != 1 && !(NF == 2 && $2 == "<NOISE>") ){ print $0 } }' \
        > data/wsjcam0_${dset}/text
    rm -f data/wsjcam0_${dset}/text.org
done


# 2. wav.scp, utt2spk, spk2utt
for dset in ${train_set_org} ${eval_sets_org}; do
    mkdir -p data/wsjcam0_${dset}
    if [ "${dset}" == si_et_2 ];then
        mic=secondary_microphone
    else
        mic=primary_microphone
    fi
    for d in ${WSJCAM0}/data/${mic}/${dset}*/*; do
        ls ${d}*/*.wv1 &> /dev/null || continue

        for ifo in ${d}/*.ifo; do
            gender=$(<${ifo} grep "Talker Sex" | cut -d' ' -f 3)
            # Get the first char, i.e. female->f, male->m
            gender=${gender:0:1}
            # Use the first found ifo file
            break
        done
        spkid=${d##*/}
        echo "${spkid} ${gender}" 1>&3

        for wav in ${d}*/*.wv1; do
            uttid=$(basename ${wav%.*})
            echo "${uttid} sph2pipe -f wav ${wav} |" 1>&4
            <${wav} echo "${uttid} ${spkid}" 1>&5
        done
    done 3>data/wsjcam0_${dset}/spk2gender.tmp 4>data/wsjcam0_${dset}/wav.scp.tmp 5>data/wsjcam0_${dset}/utt2spk.tmp

    <data/wsjcam0_${dset}/spk2gender.tmp sort -k1 >data/wsjcam0_${dset}/spk2gender
    <data/wsjcam0_${dset}/wav.scp.tmp sort -k1 >data/wsjcam0_${dset}/wav.scp
    <data/wsjcam0_${dset}/utt2spk.tmp sort -k1 >data/wsjcam0_${dset}/utt2spk
    rm -f data/wsjcam0_${dset}/*.tmp

    <data/wsjcam0_${dset}/utt2spk utils/utt2spk_to_spk2utt.pl >data/wsjcam0_${dset}/spk2utt
    utils/fix_data_dir.sh data/wsjcam0_${dset}
    utils/validate_data_dir.sh --no-feats data/wsjcam0_${dset}/
done
