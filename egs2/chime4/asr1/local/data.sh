#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=0
stop_stage=2
train_dev=dt05_multi_isolated_1ch_track
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;



if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi

if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi

if [ ! -e "${CHIME4}" ]; then
    log "Fill the value of 'CHIME4' of db.sh"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"

    wsj0_data=${CHIME4}/data/WSJ0
    local/clean_wsj0_data_prep.sh ${wsj0_data}
    local/clean_chime4_format_data.sh

    # create data for 1ch and 2ch tracks
    if [ ! -d ${CHIME4}/data/audio/16kHz/isolated_1ch_track ]; then
        log "create data for 1ch tracks"
        python local/sym_channel.py ${CHIME4} 1ch
    fi

    if [ ! -d ${CHIME4}/data/audio/16kHz/isolated_2ch_track ]; then
        log "create data for 2ch tracks"
        python local/sym_channel.py ${CHIME4} 2ch
    fi

    # beamforming for multich
    local/run_beamform_2ch_track.sh --cmd "${train_cmd}" --nj 20 \
	    ${CHIME4}/data/audio/16kHz/isolated_2ch_track enhan/beamformit_2mics
    local/run_beamform_6ch_track.sh --cmd "${train_cmd}" --nj 20 \
	    ${CHIME4}/data/audio/16kHz/isolated_6ch_track enhan/beamformit_5mics

    # preparation for chime4 data
    local/real_noisy_chime4_data_prep.sh ${CHIME4}
    local/simu_noisy_chime4_data_prep.sh ${CHIME4}

    # test data for 1ch track
    local/real_enhan_chime4_data_prep.sh isolated_1ch_track ${CHIME4}/data/audio/16kHz/isolated_1ch_track
    local/simu_enhan_chime4_data_prep.sh isolated_1ch_track ${CHIME4}/data/audio/16kHz/isolated_1ch_track

    # test data for 2ch track
    local/real_enhan_chime4_data_prep.sh beamformit_2mics ${PWD}/enhan/beamformit_2mics
    local/simu_enhan_chime4_data_prep.sh beamformit_2mics ${PWD}/enhan/beamformit_2mics

    # test data for 6ch track
    local/real_enhan_chime4_data_prep.sh beamformit_5mics ${PWD}/enhan/beamformit_5mics
    local/simu_enhan_chime4_data_prep.sh beamformit_5mics ${PWD}/enhan/beamformit_5mics

    # Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
    local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "combine real and simulation data"

    # TO DO:--extra-files but no utt2num_frames
    utils/combine_data.sh data/tr05_multi_noisy data/tr05_simu_noisy data/tr05_real_noisy
    utils/combine_data.sh data/tr05_multi_noisy_si284 data/tr05_multi_noisy data/train_si284
    utils/combine_data.sh data/${train_dev} data/dt05_simu_isolated_1ch_track data/dt05_real_isolated_1ch_track
fi

other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Srctexts preparation"

    mkdir -p "$(dirname ${other_text})"

    # NOTE(kamo): Give utterance id to each texts.
    zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
	    grep -v "<" | tr "[:lower:]" "[:upper:]" | \
	    awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}
    log "Create non linguistic symbols: ${nlsyms}"
    cut -f 2- data/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
