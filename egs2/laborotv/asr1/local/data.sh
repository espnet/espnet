#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=2
stop_stage=3

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${LABOROTV}" ]; then
    log "Fill the value of 'LABOROTV' of db.sh"
    exit 1
fi

if [ ! -e "${TEDXJP}" ]; then
    log "Fill the value of 'TEDXJP' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${TEDXJP}/-6K2nN9aWsg.wav" ]; then
        echo "stage 1: Data Download to ${TEDXJP}"
	echo "Note that the current downloaded data has some issues and we basically skip this"
	echo "stage for now."
	awk -F',' 'NR > 1 {print $3}' < local/tedx-jp/tedx-jp-10k.csv | while IFS= read -r line; do
	    n=0
	    until [ "${n}" -ge 5 ]; do
		youtube-dl \
		    --extract-audio \
		    --audio-format wav \
		    --write-sub \
		    --sub-format vtt \
		    --sub-lang ja \
		    --output "${TEDXJP}/%(id)s.%(ext)s" \
		    ${line} && break
		n=$((n+1))
		echo "Try again (${n}-th trial)"
	    done
	done
    else
	log "stage 1: ${TEDXJP}/-6K2nN9aWsg.wav is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    local/tedx-jp/10k_data_prep.sh ${TEDXJP}
    local/csj_rm_tag_sp_space.sh data/tedx-jp-10k_verbatim
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Initial normalization of the data
    local/laborotv_data_prep.sh ${LABOROTV}
    local/csj_rm_tag_sp_space.sh data/train
    local/csj_rm_tag_sp_space.sh data/dev

    # make a development set
    utils/subset_data_dir.sh --first data/train 4000 data/dev_4k
    n=$(($(wc -l < data/train/text) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
