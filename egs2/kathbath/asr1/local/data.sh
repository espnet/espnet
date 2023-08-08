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
transcript_clean_url="https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/transcripts_n2w.tar"
transcript_noisy_url="https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/noisy/transcripts_n2w.tar"

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
        echo "stage 1: Data Download to ${KATHBATH}"
        for data_url in $noisy_test_known_data_url $noisy_test_unknown_data_url $transcript_noisy_url; do
            if ! wget -P $download_data --no-check-certificate $data_url; then
                echo "$0: error executing wget $data_url"
                exit 1
            fi
            fname=${data_url##*/}
            if ! tar -C $download_data -xf $download_data/$fname; then
                echo "$0: error un-tarring archive $download_data/$fname"
                exit 1
            fi
            rm $download_data/$fname
        done
        for data_url in $clean_test_known_data_url $clean_test_unknown_data_url $train_data_url $valid_data_url $transcript_clean_url; do
            if ! wget -P $download_data --no-check-certificate $data_url; then
                echo "$0: error executing wget $data_url"
                exit 1
            fi
            fname=${data_url##*/}
            if ! tar -C $download_data -xf $download_data/$fname; then
                echo "$0: error un-tarring archive $download_data/$fname"
                exit 1
            fi
            rm $download_data/$fname
        done
    touch "${KATHBATH}/download_done"
    else
        log "stage 1: "${KATHBATH}/download_done" is already existing. Skip data downloading"
    fi
fi

mkdir -p data
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ ! -e "data/dataprep_done" ]; then
        log "stage 2: Data Preparation"
        for lang in "$download_data"/"kb_data_clean_m4a"/* ; do
            log "Processing $lang"
            for split in "$lang"/* ; do
                path=$split/"audio"

                find "$path" -type f -name "*.m4a" -exec bash -c 'ffmpeg -nostdin -loglevel warning -hide_banner -stats -i "$0" -ar 16000 -ac 1 "${0/.m4a/.wav}"' {} \;

            done
            ln=$(basename $lang)
            mkdir -p data/"$ln"
            for split in "$lang"/* ; do
                splitname=$(basename $split)
                mkdir -p data/"$ln"/"$splitname"
                find $split/"audio" -iname "*.wav" > data/"$ln"/"$splitname"/wavs ; rev data/"$ln"/"$splitname"/wavs | cut -d'/' -f1 | rev | cut -d'.' -f1 > data/"$ln"/"$splitname"/uttids ;
                paste -d'\t' data/"$ln"/"$splitname"/uttids data/"$ln"/"$splitname"/wavs > data/"$ln"/"$splitname"/wav.scp
                paste -d'\t' data/"$ln"/"$splitname"/uttids data/"$ln"/"$splitname"/uttids > data/"$ln"/"$splitname"/utt2spk
                sed "s/.m4a//g" "$split"/"transcription_n2w.txt" > data/"$ln"/"$splitname"/text
                sed 's/\xC2\xA0/ /g' data/"$ln"/"$splitname"/text > data/"$ln"/"$splitname"/temp ; mv data/"$ln"/"$splitname"/temp data/"$ln"/"$splitname"/text
                utils/utt2spk_to_spk2utt.pl data/"$ln"/"$splitname"/utt2spk > data/"$ln"/"$splitname"/spk2utt
                utils/fix_data_dir.sh data/"$ln"/"$splitname"
                rm data/"$ln"/"$splitname"/wavs data/"$ln"/"$splitname"/uttids
            done

        done

        for lang in "$download_data"/"kb_data_noisy_m4a"/* ; do
            log "Processing $lang"
            for split in "$lang"/* ; do
                log $split
                path=$split/"audio"

                find "$path" -type f -name "*.m4a" -exec bash -c 'ffmpeg -nostdin -loglevel warning -hide_banner -stats -i "$0" -ar 16000 -ac 1 "${0/.m4a/.wav}"' {} \;

            done
            ln=$(basename $lang)
            mkdir -p data/"$ln"
            for split in "$lang"/* ; do
                splitname=$(basename $split)"_noisy"
                if [ "$splitname" == "valid_noisy" ]; then
                    continue
                fi
                mkdir -p data/"$ln"/"$splitname"
                find $split/"audio" -iname "*.wav" > data/"$ln"/"$splitname"/wavs ; rev data/"$ln"/"$splitname"/wavs | cut -d'/' -f1 | rev | cut -d'.' -f1 > data/"$ln"/"$splitname"/uttids ;
                paste -d'\t' data/"$ln"/"$splitname"/uttids data/"$ln"/"$splitname"/wavs > data/"$ln"/"$splitname"/wav.scp
                paste -d'\t' data/"$ln"/"$splitname"/uttids data/"$ln"/"$splitname"/uttids > data/"$ln"/"$splitname"/utt2spk
                sed "s/.m4a//g" "$split"/"transcription_n2w.txt" > data/"$ln"/"$splitname"/text
                sed 's/\xC2\xA0/ /g' data/"$ln"/"$splitname"/text > data/"$ln"/"$splitname"/temp ; mv data/"$ln"/"$splitname"/temp data/"$ln"/"$splitname"/text
                utils/utt2spk_to_spk2utt.pl data/"$ln"/"$splitname"/utt2spk > data/"$ln"/"$splitname"/spk2utt
                utils/fix_data_dir.sh data/"$ln"/"$splitname"
                rm data/"$ln"/"$splitname"/wavs data/"$ln"/"$splitname"/uttids
            done
        done
        touch "data"/"dataprep_done"
    else
        log "stage 2: data/dataprep_done is complete"
    fi
fi
