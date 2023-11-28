#! /bin/bash 

# Apache 2.0 Roshan Sharma (Carnegie Mellon University) 2023
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

. ./db.sh


stage=1
stop_stage=100000
log "$0 $*"
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${SLUETED}" ]; then
    log "Fill the value of 'SLUETED' of db.sh"
    exit 1
fi

download_dir=${SLUETED}

train_url=https://huggingface.co/datasets/asapp/slue-phase-2/resolve/main/data/slue-ted_train.zip?download=true
dev_url=https://huggingface.co/datasets/asapp/slue-phase-2/resolve/main/data/slue-ted_dev.zip?download=true
test_url=https://huggingface.co/datasets/asapp/slue-phase-2/resolve/main/data/slue-ted_test_blind.zip?download=true


## TSV File has columns: id	transcript	speaker	split	title	abstract


# Download data 


mkdir -p ${SLUETED}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Download data to ${SLUETED}"
    [ -f ${SLUETED}/slue-ted_train.zip ] || wget -O ${SLUETED}/slue-ted_train.zip ${train_url} && unzip ${SLUETED}/slue-ted_train.zip && mv ${SLUETED}/slue-ted ${SLUETED}/slue-ted_fine-tune
    [ -f ${SLUETED}/slue-ted_dev.zip ] || wget -O ${SLUETED}/slue-ted_dev.zip ${dev_url} && unzip ${SLUETED}/slue-ted_dev.zip
    [ -f ${SLUETED}/slue-ted_test_blind.zip ] || wget -O ${SLUETED}/slue-ted_test_blind.zip ${test_url} && unzip ${SLUETED}/slue-ted_test_blind.zip && mv ${SLUETED}/slue-ted_test_blind ${SLUETED}/slue-ted_test

fi 

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Data Preparation for ${SLUETED}"

    mkdir -p data/{fine-tune,dev,test_blind}

    for partition in fine-tune dev test_blind; do 
        if  [[ "$partition" == "fine-tune"  ]]; then 
            prefix=${SLUETED}/slue-ted_fine-tune/fine-tune
        elif [[ "$partition" == "dev"  ]]; then
            prefix=${SLUETED}/slue-ted_dev/dev
        elif [[ "$partition" == "test_blind"  ]]; then
            prefix=${SLUETED}/slue-ted_test/test
        fi 
        cut -d $'\t' -f 1,3,6,5 ${prefix}/../slue-ted_${partition}.tsv | grep -v "title	abstract" | awk -F ' ' '{print $1_$2,$3,"<SEP>",$4}'  > data/${partition}/text
        cut -d $'\t' -f 1,3 ${prefix}/../slue-ted_${partition}.tsv | grep -v "id	transcript" | awk -F ' ' '{print $1_$2,$2}' > data/${partition}/utt2spk
        cut -d $'\t' -f 1,3 ${prefix}/../slue-ted_${partition}.tsv | grep -v "id	transcript" | awk -F ' ' -v x=${prefix} '{print $1_$2,x"/"$2".flac"}' > data/${partition}/wav.scp
        utils/utt2spk_to_spk2utt.pl data/${partition}/utt2spk > data/${partition}/spk2utt
    done 

fi 