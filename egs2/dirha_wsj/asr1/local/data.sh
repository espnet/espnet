#!/bin/bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=1
mic=Beam_Circular_Array # Beam_Circular_Array Beam_Linear_Array KA6 L1C

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh
. ./db.sh


if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi

if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi

if [ ! -e "${DIRHA_WSJ}" ]; then
    log "Fill the value of 'DIRHA_WSJ' of db.sh"
    exit 1
fi

if [ -z "${DIRHA_WSJ_PROCESSED}" ]; then
    log "Fill the value of 'DIRHA_WSJ_PROCESSED' of db.sh"
    exit 1
fi

if [ ! -e "${DIRHA_ENGLISH_PHDEV}" ]; then
    log "Fill the value of 'DIRHA_ENGLISH_PHDEV' of db.sh"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"

    mkdir -p "${DIRHA_WSJ_PROCESSED}"
    matlab -nodisplay -nosplash -r "addpath('./local/tools'); Data_Contamination('${mic}','${WSJ1}', '${WSJ0}', '${DIRHA_WSJ}', '${DIRHA_WSJ_PROCESSED}', '${DIRHA_ENGLISH_PHDEV}/Data/Training_IRs', 'sph2pipe');exit"

    # augmented train
    wsj0_contaminated_folder=WSJ0_contaminated_mic_${mic} # path of the wsj0 training data
    wsj1_contaminated_folder=WSJ1_contaminated_mic_${mic} # path of the wsj1 training data
    local/wsj_data_prep.sh ${DIRHA_WSJ_PROCESSED}/${wsj0_contaminated_folder}/??-{?,??}.? ${DIRHA_WSJ_PROCESSED}/${wsj1_contaminated_folder}/??-{?,??}.?
    local/wsj_format_data.sh ${mic}

    # driha test
    local/dirha_data_prep.sh ${DIRHA_WSJ_PROCESSED}/DIRHA_wsj_oracle_VAD_mic_${mic}/Sim dirha_sim_${mic}
    local/dirha_data_prep.sh ${DIRHA_WSJ_PROCESSED}/DIRHA_wsj_oracle_VAD_mic_${mic}/Real dirha_real_${mic}
fi

other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Srctexts preparation"

    mkdir -p "$(dirname ${other_text})"

    # NOTE(kamo): Give utterance id to each texts.
    zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
	    grep -v "<" | tr "[:lower:]" "[:upper:]" | \
	    awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}

    log "Create non linguistic symbols: ${nlsyms}"
    cut -f 2- data/train_si284_"${mic}"/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
