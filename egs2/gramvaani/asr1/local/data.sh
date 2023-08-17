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

train_data_url="https://asr.iitm.ac.in/Gramvaani/NEW/GV_Train_100h.tar.gz"
dev_data_url="https://asr.iitm.ac.in/Gramvaani/NEW/GV_Dev_5h.tar.gz"
test_data_url="https://asr.iitm.ac.in/Gramvaani/NEW/GV_Eval_3h.tar.gz"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${GRAMVAANI}" ]; then
    log "Fill the value of 'GRAMVAANI' of db.sh"
    exit 1
fi

download_data="${GRAMVAANI}"
mkdir -p $download_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e $download_data/"downloads_done" ]; then
        echo "stage 1: Data Download to $download_data"
        for data_url in $train_data_url $dev_data_url $test_data_url; do
            if ! wget -P $download_data --no-check-certificate $data_url; then
                echo "$0: error executing wget $data_url"
                exit 1
            fi
            fname=${data_url##*/}
            if ! tar -C $download_data -xvzf $download_data/$fname; then
                echo "$0: error un-tarring archive $download_data/$fname"
                exit 1
            fi
            rm $download_data/$fname
        done
        touch $download_data/"downloads_done"
    else
        log "stage 1: ${GRAMVAANI}/downloads_done is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ ! -e "data/dataprep_done" ]; then
        log "stage 2: Data Preparation"
        mkdir -p data
        mkdir -p data/{train100,dev,test}

        cp $download_data/"GV_Train_100h"/{mp3.scp,text} data/train100
        cp $download_data/"GV_Dev_5h"/{mp3.scp,text} data/dev
        cp $download_data/"GV_Eval_3h"/{mp3.scp,text} data/test
        sed "s:./:$download_data/GV_Eval_3h/:" data/test/mp3.scp > data/temp ; mv data/temp data/test/mp3.scp
        sed "s:./:$download_data/GV_Dev_5h/:" data/dev/mp3.scp > data/temp ; mv data/temp data/dev/mp3.scp
        sed "s:./:$download_data/GV_Train_100h/:" data/train100/mp3.scp > data/temp ; mv data/temp data/train100/mp3.scp

        for x in train100 dev test; do

            fname="data/"$x/"text"
            python3 local/clean_text.py $fname
            cut -d' ' -f1 data/"$x"/text > data/clean_ids
            grep -f data/clean_ids data/"$x"/mp3.scp > data/temp ; mv data/temp data/"$x"/mp3.scp
            paste -d" " "data/clean_ids" "data/clean_ids" > "data/"$x/"utt2spk"
            rm data/clean_ids

            savepath="data/"$x/"wav.scp"
            savefolder=$download_data/"resampled"
            mkdir -p $savefolder
            savefolder=$savefolder"/"$x
            mkdir -p $savefolder
            [ -e $savepath ] && rm $savepath
            while IFS= read -r line;do
                path=$(echo $line | cut -d' ' -f2)
                id=$(echo $line | cut -d' ' -f1)
                savename="$savefolder"/"$id".wav
                sox $path -c 1 $savename rate 16000
                echo $id$" "$savename >> $savepath
                done < "data/"$x/"mp3.scp"

            utils/utt2spk_to_spk2utt.pl <"data/"$x/"utt2spk" >"data/"$x/"spk2utt"
            utils/fix_data_dir.sh "data/"$x

        done
        touch data/dataprep_done
    else
        log "stage 2: data/dataprep_done is already existing. Skip data prep"
    fi
fi
