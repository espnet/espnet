#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 --extra-annotations <path> [--stage <stage>] [--stop_stage <stop_stage>] [--nj <nj>]

  required argument:
    --extra-annotations: path to a directory containing extra annotations for CHiME4
                         This is required for preparing et05_simu_isolated_1ch_track.
    NOTE:
        You can download it manually from
            http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME4/download.html
        Then unzip the downloaded file to CHiME4_diff;
        You will then find the extra annotations in CHiME4_diff/CHiME3/data/annotations

  optional argument:
    [--stage]: 1 (default) or 2
    [--stop_stage]: 1 or 2 (default)
    [--nj]: number of parallel pool workers in MATLAB
EOF
)


nj=10
stage=1
stop_stage=2
extra_annotations=
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ] || [ -z "${extra_annotations}" ]; then
    echo "${help_message}"
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


odir="${PWD}/local/nn-gev/data"; mkdir -p "${odir}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Simulation"

    # if ! command -v matlab &> /dev/null; then
    #     log "You don't have matlab"
    #     exit 2
    # fi

    # Prepare simulation data for 6ch track
    # (This takes ~10 hours with nj=10 on Intel(R) Xeon(R) CPU E5-2670 v2 @ 2.50GHz)
    # Expected data directories to be generated (~40 GB):
    #   - ${odir}/audio/16kHz/isolated_ext/*/*.CH?.{Clean,Noise}.wav
    #   - ${odir}/audio/16kHz/isolated/*/*.CH?.wav
    # -----------------------------------------------------------------------------------------------
    # directory                   disk usage  duration      #samples
    # -----------------------------------------------------------------------------------------------
    # isolated_ext/tr05_bus_simu  4.9 GB      44h 29m 45s   1728 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/tr05_caf_simu  5.0 GB      45h 17m 23s   1794 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/tr05_ped_simu  4.9 GB      44h 58m 57s   1765 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/tr05_str_simu  5.1 GB      46h 59m 49s   1851 * 6 * 2 (6 channels, Clean & Noise)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # isolated_ext/dt05_bus_simu  964 MB      8h 40m 50s    410 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/dt05_caf_simu  964 MB      8h 40m 50s    410 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/dt05_ped_simu  964 MB      8h 40m 50s    410 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/dt05_str_simu  964 MB      8h 40m 50s    410 * 6 * 2 (6 channels, Clean & Noise)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # isolated_ext/et05_bus_simu  760 MB      6h 50m 18s    330 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/et05_caf_simu  760 MB      6h 50m 18s    330 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/et05_ped_simu  760 MB      6h 50m 18s    330 * 6 * 2 (6 channels, Clean & Noise)
    # isolated_ext/et05_str_simu  760 MB      6h 50m 18s    330 * 6 * 2 (6 channels, Clean & Noise)
    # -----------------------------------------------------------------------------------------------
    # isolated/tr05_bus_simu      2.5 GB      22h 14m 52s   1728 * 6 (6 channels, Noisy)
    # isolated/tr05_caf_simu      2.5 GB      22h 38m 41s   1794 * 6 (6 channels, Noisy)
    # isolated/tr05_ped_simu      2.5 GB      22h 29m 28s   1765 * 6 (6 channels, Noisy)
    # isolated/tr05_str_simu      2.6 GB      23h 29m 54s   1851 * 6 (6 channels, Noisy)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # isolated/dt05_bus_simu      482 MB      4h 20m 25s    410 * 6 (6 channels, Noisy)
    # isolated/dt05_caf_simu      482 MB      4h 20m 25s    410 * 6 (6 channels, Noisy)
    # isolated/dt05_ped_simu      482 MB      4h 20m 25s    410 * 6 (6 channels, Noisy)
    # isolated/dt05_str_simu      482 MB      4h 20m 25s    410 * 6 (6 channels, Noisy)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # isolated/et05_bus_simu      380 MB      3h 25m 9s     330 * 6 (6 channels, Noisy)
    # isolated/et05_caf_simu      380 MB      3h 25m 9s     330 * 6 (6 channels, Noisy)
    # isolated/et05_ped_simu      380 MB      3h 25m 9s     330 * 6 (6 channels, Noisy)
    # isolated/et05_str_simu      380 MB      3h 25m 9s     330 * 6 (6 channels, Noisy)
    # -----------------------------------------------------------------------------------------------

    log "Generating simulation data and storing in ${odir}"
    ${train_cmd} $odir/simulation.log matlab -nodisplay -nosplash -r "addpath('local'); CHiME3_simulate_data_patched_parallel(1,$nj,'${CHIME4}','${CHIME3}');exit"

    # Validate data simulation
    num_wavs=$(find "$odir/audio/16kHz/isolated" -iname "*.wav" | wc -l)
    if [ "$num_wavs" != "60588" ]; then
        log "Error: Expected 60588 wav files in '$odir/audio/16kHz/isolated', but got $num_wavs"
        exit 1
    fi
    num_wavs=$(find "$odir/audio/16kHz/isolated_ext" -iname "*.wav" | wc -l)
    if [ "$num_wavs" != "121176" ]; then
        log "Error: Expected 121176 wav files in '$odir/audio/16kHz/isolated_ext', but got $num_wavs"
        exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"

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

    # prepare data for 1ch track: (TODO: et05_simu_isolated_1ch_track)
    #  (1) {tr05,dt05,et05}_simu_isolated_1ch_track
    local/simu_ext_chime4_data_prep.sh --track 1 --annotations ${CHIME4}/data/annotations \
        --extra-annotations ${extra_annotations} isolated_1ch_track ${odir}/audio/16kHz
    #  (2) {tr05,dt05,et05}_real_isolated_1ch_track (TODO: tr05_real_isolated_1ch_track)
    local/real_ext_chime4_data_prep.sh --track 1 --isolated_6ch_dir ${CHIME4}/data/audio/16kHz/isolated_6ch_track \
        isolated_1ch_track ${CHIME4}/data/audio/16kHz/isolated_1ch_track

    # prepare data for 6ch track:
    #  (1) {tr05,dt05,et05}_simu_isolated_6ch_track
    local/simu_ext_chime4_data_prep.sh --track 6 isolated_6ch_track ${odir}/audio/16kHz
    #  (2) {tr05,dt05,et05}_real_isolated_6ch_track
    local/real_ext_chime4_data_prep.sh --track 6 isolated_6ch_track ${CHIME4}/data/audio/16kHz/isolated_6ch_track
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
