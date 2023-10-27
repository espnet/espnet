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
stop_stage=100000
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

#data_dict={"LIBRISPEECH":"asr", "LIBRITTS":"tts", "VCTK":"tts", "LIBRISPEECH":"lm", "LIBRILIGHT":"lm"}

declare -a arr=("librispeech" "libritts" "vctk" "librilight")
declare -a arr_task=("asr" "tts" "textlm" "speechlm")

data_root="data" #/temp"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for dataset in "${arr[@]}"
    do
        echo $dataset
        if ! [ -f local/data_${dataset}.sh ]; then
            echo "File local/data_${dataset}.sh does not exist."
            exit 1
        else
            ./local/data_${dataset}.sh "${data_root}/${dataset}"
        fi
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    
    # Combine all tasks into "train/ dev/ test"
    for _dset in "train" "dev" "test"; do
        for task in "${arr_task[@]}"; do
            speech_dirs=""
            text_data_paths=""
            for dataset in "${arr[@]}"; do
                # check if dataset-task pair exists
                src_path=${data_root}/${dataset}/${task}/${_dset}
                # echo "Checking $src_path"
                if [ -d "$src_path" ]; then
                    # text
                    if [ -f "${src_path}/text" ] &&  [ ! -f "${src_path}/wav.scp" ]; then
                        text_data_paths+=" ${src_path}/text"
                    fi

                    # speech
                    if [ -f "${src_path}/wav.scp" ]; then
                        speech_dirs+=" ${src_path}"
                        utils/validate_data_dir.sh --no-feats ${src_path}
                    fi
                fi
            done

            # combine speech dirs
            if [ ! -z "${speech_dirs}" ]; then
                utils/combine_data.sh  data/${_dset}/speech/${task} ${speech_dirs}
            fi
            
            # combine text
            if [ ! -z "${text_data_paths}" ]; then
                mkdir -p data/${_dset}/text/${task}
                echo "${_dset}: Combine text: "${text_data_paths}
                for f in ${text_data_paths}; do
                    cat "${f}"
                done>"data/${_dset}/text/${task}/text"
            fi
        done

    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
