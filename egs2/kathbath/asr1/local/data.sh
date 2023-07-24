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

train_data_url="https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/train_audio.tar"
valid_data_url="https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/valid_audio.tar"
clean_test_known_data_url="https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/testkn_audio.tar"
clean_test_unknown_data_url="https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/testunk_audio.tar"
noisy_test_known_data_url="https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/noisy/testkn_audio.tar"
noisy_test_unknown_data_url="https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/noisy/testunk_audio.tar"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${KATHBATH}" ]; then
    log "Fill the value of 'KATHBATH' of db.sh"
    exit 1
fi

download_data="${KATHBATH}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${KATHBATH}/download_done" ]; then
        echo "stage 1: Data Download to ${LIBRISPEECH}"
        
        for data_url in $noisy_test_known_data_url $noisy_test_unknown_data_url; do
            # if ! wget -P $download_data --no-check-certificate $data_url; then
            #     echo "$0: error executing wget $data_url"
            #     exit 1
            # fi
            fname=${data_url##*/}
            # savefolder=$download_data/"noisy_"$(echo $fname | cut -f 1 -d '.')
            # mkdir -p $savefolder
            # if ! tar -C $savefolder -xf $download_data/$fname; then
            #     echo "$0: error un-tarring archive $download_data/$fname"
            #     exit 1
            # fi
            echo "noisy data is not downloaded"
            # rm $download_data/$fname
        done
        for data_url in $clean_test_known_data_url $clean_test_unknown_data_url; do
            # if ! wget -P $download_data --no-check-certificate $data_url; then
            #     echo "$0: error executing wget $data_url"
            #     exit 1
            # fi
            # fname=${data_url##*/}
            # savefolder=$download_data/"clean_"$(echo $fname | cut -f 1 -d '.')
            # mkdir -p $savefolder
            # if ! tar -C $savefolder -xf $download_data/$fname; then
            #     echo "$0: error un-tarring archive $download_data/$fname"
            #     exit 1
            # fi
            # rm $download_data/$fname
        done
        for data_url in $train_data_url $valid_data_url; do
            # if ! wget -P $download_data --no-check-certificate $data_url; then
            #     echo "$0: error executing wget $data_url"
            #     exit 1
            # fi
            fname=${data_url##*/}
            if ! tar -C $download_data -xf $download_data/$fname; then
                echo "$0: error un-tarring archive $download_data/$fname"
                exit 1
            fi
            rm $download_data/$fname
        done

    # touch "${KATHBATH}/download_done"
    else
        log "stage 1: "${KATHBATH}/download_done" is already existing. Skip data downloading"
    fi
fi
