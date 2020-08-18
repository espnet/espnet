#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


nj=10
stage=0
stop_stage=2
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ! -e "${CHIME3}" ]; then
    log "Fill the value of 'CHIME3' in db.sh"
    exit 1
fi

if [ ! -e "${CHIME4}" ]; then
    log "Fill the value of 'CHIME4' in db.sh"
    exit 1
fi


if [${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Simulation"

    # prepare simulation data for 6ch track:
    log "Generating simulation data and storing in local/nn-gev/data"
    odir=local/nn-gev/data; mkdir $odir
    ${train_cmd} $odir/simulation.log matlab -nodisplay -nosplash -r "addpath('local'); CHiME3_simulate_data_patched_parallel(1,$nj,'${CHIME4}','${CHIME3}');exit"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"
    
    # preparation for original WSJ0 data:
    #  et05_orig_clean, dt05_orig_clean, tr05_orig_clean
    wsj0_data=${CHIME4}/data/WSJ0
    local/clean_wsj0_data_prep.sh ${wsj0_data}
    local/clean_chime4_format_data.sh

    # preparation for chime4 data:
    #  (1) tr05_real_noisy, dt05_real_noisy, et05_real_noisy
    local/real_noisy_chime4_data_prep.sh ${CHIME4}
    #  (2) tr05_simu_noisy, dt05_simu_noisy, et05_simu_noisy
    local/simu_noisy_chime4_data_prep.sh ${CHIME4}

    # prepare data for 1ch track:
    #  (1) {tr05,dt05,et05}_simu_isolated_1ch_track
    local/simu_ext_chime4_data_prep.sh --track 1 --annotations ${CHIME4}/data/annotations isolated_1ch_track ${PWD}/local/nn-gev/data/audio/16kHz
    #  (2) {tr05,dt05,et05}_real_isolated_1ch_track
    local/real_ext_chime4_data_prep.sh --track 1 isolated_1ch_track ${CHIME4}/data/audio/16kHz/isolated_1ch_track

    # prepare data for 6ch track:
    #  (1) {tr05,dt05,et05}_simu_isolated_6ch_track
    local/simu_ext_chime4_data_prep.sh --track 6 isolated_6ch_track ${PWD}/local/nn-gev/data/audio/16kHz
    #  (2) {tr05,dt05,et05}_real_isolated_6ch_track
    local/real_ext_chime4_data_prep.sh --track 6 isolated_6ch_track ${CHIME4}/data/audio/16kHz/isolated_6ch_track
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	log "combine real and simulation data"

    # 1-ch track
	utils/combine_data.sh data/tr05_multi_isolated_1ch_track data/tr05_simu_isolated_1ch_track data/tr05_real_isolated_1ch_track --extra_files spk1.scp
	utils/combine_data.sh data/dt05_multi_isolated_1ch_track data/dt05_simu_isolated_1ch_track data/dt05_real_isolated_1ch_track --extra_files spk1.scp
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
