#!/bin/bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh
. ./path.sh
. ./cmd.sh


stage=1
stop_stage=2
wdir=data/local
ref_mic=LA6 # Select here one of the available microphone (e.g., LA6, L1R, LD07, Beam_Circular_Array,Beam_Linear_Array, etc. => Please, see Floorplan)

. utils/parse_options.sh


if [ ! -e "${TIMIT}" ]; then
    log "Fill the value of 'TIMIT' of db.sh"
    exit 1
fi
if [ ! -e "${DIRHA}" ]; then
    log "Fill the value of 'DIRHA' of db.sh"
    log "(You can download the dataset from here: https://dirha.fbk.eu/dirha-english-phdev-agreement)"
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Simulation"
    # Following the instructions in https://github.com/SHINE-FBK/DIRHA_English_phrich
    #  to generate training and test data (timit data contaminated with noise and reverberation)
    if ! command -v matlab >/dev/null 2>&1; then
        echo "matlab not found."
        exit 1
    fi

    mkdir -p ${wdir}/Tools
    matlab_version=$(matlab -r quit -nojvm | grep -Poi "(?<=R)201[0-9]")
    if [ $matlab_version -ge 2014 ]; then
        url=https://raw.githubusercontent.com/SHINE-FBK/DIRHA_English_phrich/master/Tools

        for fname in Data_Contamination.m create_folder_str.m find_files.m linear_shift.m; do
            wget --continue -O ${wdir}/Tools/${fname} ${url}/${fname}
        done

        sed -i -e "s#timit_folder='/path/to/TIMIT';#timit_folder='${TIMIT}';#" \
            -e "s#out_folder='../Data';#out_folder='${PWD}/data/Data';#" \
            -e "s#'../Data/#'${PWD}/data/Data/#" \
            -e "s#mic_sel='LA6';#mic_sel='${ref_mic}';#" \
            ${wdir}/Tools/Data_Contamination.m

        matlab_cmd="matlab -nodesktop -nodisplay -nosplash -r Data_Contamination"
    else
        url=https://raw.githubusercontent.com/SHINE-FBK/DIRHA_English_phrich/master/Tools/for_Matlab_older_than_R2014a

        for fname in OldMatlab_Data_Contamination.m create_folder_str.m find_files.m linear_shift.m readsph.m; do
            wget --continue -O ${wdir}/Tools/${fname} ${url}/${fname}
        done

        sed -i -e "s#timit_folder='/nfsmnt/shinefs0/data/misc/TIMIT';#timit_folder='${TIMIT}';#" \
            -e "s#out_folder='../../Data';#out_folder='${PWD}/data/Data';#" \
            -e "s#'../../Data/#'${PWD}/data/Data/#" \
            -e "s#mic_sel='LA6';#mic_sel='${ref_mic}';#" \
            ${wdir}/Tools/OldMatlab_Data_Contamination.m

        matlab_cmd="matlab -nodesktop -nodisplay -nosplash -r OldMatlab_Data_Contamination"
    fi


    cmdfile=$(realpath ${wdir}/Tools/contaminate_timit.sh)
    echo "#!/bin/bash" > $cmdfile
    echo $matlab_cmd >> $cmdfile
    chmod +x $cmdfile

    # Run Matlab (This takes more than 8 hours)
    # Expected data directories to be generated:
    #   - data/Data/Training_IRs/rir_*.mat
    #   - data/Data/TIMIT_noise_sequences/*.wav
    #   - data/Data/TIMIT_revnoise_mic/wav16k/${min_or_max}/{tr,cv,tt}/{mix,s1,s2}/*.wav
    (
    cd ${wdir}/Tools && \
    echo "Log is in ${wdir}/Tools/contaminate_timit.log" && \
    $train_cmd contaminate_timit.log $cmdfile
    )
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"


fi