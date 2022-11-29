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


stage=0
stop_stage=100
extra_annotations=
local_data_opts=
train_dev=dt05_multi_isolated_1ch_track
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ] || [ -z "${extra_annotations}" ]; then
    echo "${help_message}"
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Enh data preparation"
    local/chime4_enh_data.sh --extra_annotations ${extra_annotations} ${local_data_opts}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: ASR data preparation"
    local/chime4_asr_data.sh --stage 0 --stop-stage 1 ${local_data_opts}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Enh_ASR data preparation: combine enh and asr data"

    # dummy spk1.scp
    for dset in tr05_real_noisy train_si284 dt05_real_isolated_1ch_track et05_real_isolated_1ch_track dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics; do
        cp data/${dset}/wav.scp data/${dset}/spk1.scp
    done
    cp data/tr05_simu_isolated_1ch_track/spk1.scp data/tr05_simu_noisy

    # utt2category
    <data/tr05_simu_noisy/wav.scp awk '{print($1, "SIMU")}' > data/tr05_simu_noisy/utt2category
    <data/tr05_real_noisy/wav.scp awk '{print($1, "REAL")}' > data/tr05_real_noisy/utt2category
    <data/train_si284/wav.scp awk '{print($1, "CLEAN")}' > data/train_si284/utt2category
    <data/dt05_simu_isolated_1ch_track/wav.scp awk '{print($1, "SIMU")}' > data/dt05_simu_isolated_1ch_track/utt2category
    <data/dt05_real_isolated_1ch_track/wav.scp awk '{print($1, "REAL")}' > data/dt05_real_isolated_1ch_track/utt2category

    utils/combine_data.sh --extra_files "utt2category spk1.scp" \
        data/tr05_multi_noisy data/tr05_simu_noisy data/tr05_real_noisy 
    utils/combine_data.sh --extra_files "utt2category spk1.scp" \
        data/tr05_multi_noisy_si284 data/tr05_multi_noisy data/train_si284
    utils/combine_data.sh --extra_files "utt2category spk1.scp" data/${train_dev} \
        data/dt05_simu_isolated_1ch_track data/dt05_real_isolated_1ch_track

    <data/tr05_simu_isolated_6ch_track/wav.scp awk '{print($1, "SIMU")}' > data/tr05_simu_isolated_6ch_track/utt2category
    <data/tr05_real_isolated_6ch_track/wav.scp awk '{print($1, "REAL")}' > data/tr05_real_isolated_6ch_track/utt2category
    <data/dt05_simu_isolated_6ch_track/wav.scp awk '{print($1, "SIMU")}' > data/dt05_simu_isolated_6ch_track/utt2category
    <data/dt05_real_isolated_6ch_track/wav.scp awk '{print($1, "REAL")}' > data/dt05_real_isolated_6ch_track/utt2category

    utils/combine_data.sh --extra_files "utt2category spk1.scp" \
        data/tr05_multi_isolated_6ch_track data/tr05_simu_isolated_6ch_track data/tr05_real_isolated_6ch_track
    utils/combine_data.sh --extra_files "utt2category spk1.scp" \
        data/dt05_multi_isolated_6ch_track data/dt05_simu_isolated_6ch_track data/dt05_real_isolated_6ch_track
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Srctexts preparation"
    local/chime4_asr_data.sh --stage 2 --stop-stage 2

    for dset in data/*; do
        if [ -e "${dset}/text" ] && [ ! -e "${dset}/text_spk1" ]; then
            ln -s text ${dset}/text_spk1
        fi
    done
fi