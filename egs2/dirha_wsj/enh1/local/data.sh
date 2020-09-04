#!/bin/bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
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
# Select the reference microphone for the generated databases
# (e.g., LA6, L1R, LD07, Beam_Circular_Array,Beam_Linear_Array, etc.)
# => See ${DIRHA}/Additional_info/Floorplan/*.png for the complete list.
ref_mic=LA6
timit_uppercased=true # whether wav file names in the TIMIT directory are uppercased

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
    # Following the instructions in https://github.com/SHINE-FBK/DIRHA_English_wsj
    #  to generate training and test data (wsj data contaminated with noise and reverberation)
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
            -e "s#noise_folder='../Data/TIMIT_noise_sequences';#noise_folder='${DIRHA}/Data/TIMIT_noise_sequences';#" \
            -e "s#IR_folder{1}='../Data/Training_IRs/T1_O6';#IR_folder{1}='${DIRHA}/Data/Training_IRs/T1_O6';#" \
            -e "s#IR_folder{2}='../Data/Training_IRs/T2_O5';#IR_folder{2}='${DIRHA}/Data/Training_IRs/T2_O5';#" \
            -e "s#IR_folder{3}='../Data/Training_IRs/T3_O3';#IR_folder{3}='${DIRHA}/Data/Training_IRs/T3_O3';#" \
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
            -e "s#noise_folder='../../Data/TIMIT_noise_sequences';#noise_folder='${DIRHA}/Data/TIMIT_noise_sequences';#" \
            -e "s#IR_folder{1}='../../Data/Training_IRs/T1_O6';#IR_folder{1}='${DIRHA}/Data/Training_IRs/T1_O6';#" \
            -e "s#IR_folder{2}='../../Data/Training_IRs/T2_O5';#IR_folder{2}='${DIRHA}/Data/Training_IRs/T2_O5';#" \
            -e "s#IR_folder{3}='../../Data/Training_IRs/T3_O3';#IR_folder{3}='${DIRHA}/Data/Training_IRs/T3_O3';#" \
            -e "s#mic_sel='LA6';#mic_sel='${ref_mic}';#" \
            ${wdir}/Tools/OldMatlab_Data_Contamination.m

        matlab_cmd="matlab -nodesktop -nodisplay -nosplash -r OldMatlab_Data_Contamination"
    fi

    if $timit_uppercased; then
        ext=".WAV"
        # Fix problems caused by uppercased TIMIT wav names
        sed -i -e "s#list=find_files(timit_folder,'.wav');#list=find_files(timit_folder,'.WAV');#" \
            -e "s#phn_or=strrep(list{i},'.wav','.phn');#phn_or=strrep(list{i},'.WAV','.PHN');#" \
            -e "s#name_phn=strrep(name_wav,'.wav','.phn');#name_phn=strrep(name_wav,'.WAV','.PHN');#" \
            -e "s#noise_file=strrep(list{i},timit_folder,noise_folder);#noise_file=strcat(noise_folder, lower(strrep(list{i}, timit_folder, '')));#" \
            ${wdir}/Tools/Data_Contamination.m
    else
        ext=".wav"
    fi

    cmdfile=$(realpath ${wdir}/Tools/contaminate_timit.sh)
    echo "#!/bin/bash" > $cmdfile
    echo $matlab_cmd >> $cmdfile
    chmod +x $cmdfile

    # Run Matlab (This takes about 30 minutes with ref_mic=LA6)
    # Expected data directories to be generated (~ 711MB in total):
    #   - data/Data/TIMIT_revnoise_mic_${ref_mic}/test/dr[1-8]/*/*.{wav,phn} (1680 utterances)
    #   - data/Data/TIMIT_revnoise_mic_${ref_mic}/train/dr[1-8]/*/*.{wav,phn} (4620 utterances)
    (
    cd ${wdir}/Tools && \
    echo "Log is in ${wdir}/Tools/contaminate_timit.log" && \
    $train_cmd contaminate_timit.log $cmdfile
    )

    # Validate simulation is successfully finished
    num_wavs=$(find ${PWD}/data/Data/TIMIT_revnoise_mic_${ref_mic} -type f -name "*${ext}" | wc -l)
    if [ $num_wavs -ne 6300 ]; then
        log "Error: Simulation failed! See ${wdir}/Tools/contaminate_timit.log for more information"
        exit 1;
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"

    local/timit_data_prep.sh ${TIMIT} || exit 1

    local/dirha_data_prep.sh ${DIRHA}/Real ${ref_mic} 'dirha_real' || exit 1

    local/timit_prepare_dict.sh

    utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
    data/local/dict "sil" data/local/lang_tmp data/lang

    local/timit_format_data.sh
fi
