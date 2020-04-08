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

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


dirha_folder=/export/b18/ruizhili/data
WSJ0=/export/corpora5/LDC/LDC93S6B
WSJ1=/export/corpora5/LDC/LDC94S13B
dirha_wsj_folder=/export/b18/ruizhili/data/Data_processed
#IR_folder=/export/b18/xwang/data/ # folders for Impulse responses for WSJ contamination
#sph_reader=${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe

# TO DO: fix IR_folder, as well as uncomment matlab preprocess code in stage 0
# You only need dirha_wsj_folder, just forget WSJ0/1, dirha_folder for now

if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi

if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi

if [ ! -e "${dirha_folder}" ]; then
    log "Fill the value of 'dirha_folder' of db.sh"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"
    
    #expdir=exp/prepare_dirha_wsj_data_${mic}
    #$train_cmd $expdir/Data.log \
    #matlab -nodisplay -nosplash -r "addpath('./local/tools'); Data_Contamination('$mic','$WSJ0', '$WSJ1', '$dirha_folder', '$dirha_wsj_folder', '$IR_folder', '$sph_reader');exit"
    
    # augmented train
    wsj0_contaminated_folder=WSJ0_contaminated_mic_$mic # path of the wsj0 training data
    wsj1_contaminated_folder=WSJ1_contaminated_mic_$mic # path of the wsj1 training data
    local/wsj_data_prep.sh ${dirha_wsj_folder}/$wsj0_contaminated_folder/??-{?,??}.? ${dirha_wsj_folder}/$wsj1_contaminated_folder/??-{?,??}.? || exit 1;
    local/wsj_format_data.sh $mic || exit 1;

    # driha test
    DIRHA_wsj_data=${dirha_wsj_folder}/DIRHA_wsj_oracle_VAD_mic_$mic # path of the test data
    local/dirha_data_prep.sh $DIRHA_wsj_data/Sim dirha_sim_$mic  || exit 1;
    local/dirha_data_prep.sh $DIRHA_wsj_data/Real dirha_real_$mic  || exit 1;
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
    cut -f 2- data/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
