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
stop_stage=1000

# Dereverberation Measures
compute_se=true # flag for turing on computation of dereverberation measures
enable_pesq=false # please make sure that you or your institution have the license to report PESQ before turning on this flag
nch_se=8

help_message=$(cat << EOF
Usage: $0

Options:
    --compute_se (bool): Default ${compute_se}
        flag for turing on computation of dereverberation measures
    --enable_pesq (bool): Default ${enable_pesq}
        please make sure that you or your institution have the license to report PESQ before turning on this flag
    --nch_sq (int): Default ${nch_se}

EOF
)


log "$0 $*"
. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 0 ]; then
  log "${help_message}"
  exit 2
fi


# data
train_set=tr_simu_8ch_si284
train_set2=tr_wsjcam0_si284
train_dev=dt_mult_1ch
other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: REVERB Data preparation"

    if [ ! -f "${REVERB_OUT}"/.done ]; then
        log "You haven't created REVERB data. Now Trying to create it."
        rm -rf "${REVERB_OUT}"/.done

        if ! command -v matlab &> /dev/null; then
            log "You don't have matlab"
            exit 2
        fi

        if ! command -v BeamformIt &> /dev/null; then
            log "Error: You don't have BeamformIt "
            log "cd ../../../tools; installers/install_beamformit.sh"
            exit 2
        fi

        if [ -z "${WSJCAM0}" ]; then
            log "Error: \$WSJCAM0 is not set in db.sh."
            exit 2
        fi
        if [ -z "${REVERB}" ]; then
            log "Error: \$REVERB is not set in db.sh."
            exit 2
        fi
        local/generate_data.sh --wavdir "${REVERB_OUT}" "${WSJCAM0}"
        local/prepare_simu_data.sh --wavdir "${REVERB_OUT}" "${REVERB}" "${WSJCAM0}"
        local/prepare_real_data.sh --wavdir "${REVERB_OUT}" "${REVERB}"

        # Run WPE and Beamformit
        local/run_wpe.sh
        local/run_beamform.sh "${REVERB_OUT}/WPE/"
        if "${compute_se}"; then
            if [ ! -d local/REVERB_scores_source ] || [ ! -d local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/SRMRToolbox ] || [ ! -f local/PESQ ]; then
                # download and install speech enhancement evaluation tools
                local/download_se_eval_tool.sh
            fi
            pesqdir="${PWD}/local"
            local/compute_se_scores.sh --nch "${nch_se}" --enable_pesq "${enable_pesq}" "${REVERB}" "${REVERB_OUT}" "${pesqdir}"
            cat "exp/compute_se_${nch_se}ch/scores/score_SimData"
            cat "exp/compute_se_${nch_se}ch/scores/score_RealData"
        fi

        touch "${REVERB_OUT}"/.done
    else
        log "You have ${REVERB_OUT}/.done. Skipping REVERB data preparation"
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained"

    if [ -z "${WSJ0}" ]; then
        log "Error: \$WSJ0 is not set in db.sh."
        exit 2
    fi
    if [ -z "${WSJ1}" ]; then
        log "Error: \$WSJ1 is not set in db.sh."
        exit 2
    fi
    local/wsj_data_prep.sh "${WSJ0}"/??-{?,??}.? "${WSJ1}"/??-{?,??}.?
    local/wsj_format_data.sh
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Combine data"

    echo "combine reverb simulation and wsj clean training data"
    utils/combine_data.sh data/"${train_set}" data/tr_simu_8ch data/train_si284
    echo "combine real and simulation development data"
    utils/combine_data.sh data/"${train_dev}" data/dt_real_1ch data/dt_simu_1ch
fi



if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Prepare other text"
    # NOTE(kamo): Give utterance id to each texts.
    mkdir -p "$(dirname ${other_text})"
    zcat "${WSJ1}"/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
        grep -v "<" | tr "[:lower:]" "[:upper:]" | \
        awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}



    log "Create non linguistic symbols: ${nlsyms}"
    mkdir -p "$(dirname ${nlsyms})"
    cut -f 2- data/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > "${nlsyms}"
fi


# NOTE(kamo): Stage5 and Stage6 creates clean data, RIR, Noise data.
# RIR and Noise are applied in training process on-the fly
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Prepare WSJCAM0 data"
    local/prepare_wsjcam0.sh "${WSJCAM0}"
    utils/combine_data.sh data/"${train_set2}" data/wsjcam0_si_tr data/train_si284
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Prepare REVERB RIR and Noise data"
    local/prepare_rir_noise_1ch.sh
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
