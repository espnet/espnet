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

help_message=$(cat << EOF
Usage: $0 [--use_devkit_subsets <true/false>] [--stage <stage>] [--stop_stage <stop_stage>]

  optional argument:
    [--use_devkit_subsets]: true or false (default)
        whether to only use the devkit subsets or use the full datasets
    [--stage]: 1 (default) or 2
    [--stop_stage]: 1 or 2 (default)
EOF
)


stage=1
stop_stage=100000
use_devkit_subsets=false

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 2
fi


train_set="train"
valid_set="dev"
test_set="test"
if $use_devkit_subsets; then
    train_set=devkit_${train_set}
    valid_set=devkit_${valid_set}
    test_set=devkit_${test_set}
    nch=2
else
    train_set=full_${train_set}
    valid_set=full_${valid_set}
    test_set=full_${test_set}
    # Rooms 1 and 2 have 12 mics, and rooms 3 and 4 have 20 mics.
    nch=multi
fi


if [ -z "${VOICES}" ]; then
    log "Fill the value of 'VOICES' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if $use_devkit_subsets; then
        num_audios=$(find "${VOICES}"/VOiCES_devkit/ -iname "*.wav" | wc -l)
        num_tgt=20248
    else
        num_audios=$(find "${VOICES}"/VOiCES_rebuilt/ -iname "*.wav" | wc -l)
        num_tgt=1003519
    fi
    if [ $num_audios -ne $num_tgt ]; then
        echo "stage 1: Downloading data to ${VOICES}"
        if ! command -v aws &> /dev/null; then
            log "Error: aws-cli is not installed."
            log "Please follow the instruction in https://docs.aws.amazon.com/en_us/cli/latest/userguide/getting-started-install.html to install aws-cli manually."
            exit 1
        fi
        if $use_devkit_subsets; then
            lst=("VOiCES_devkit.tar.gz" "VOiCES_competition.tar.gz" "recording_data.tar.gz")
        else
            lst=("VOiCES_release.tar.gz" "VOiCES_competition.tar.gz" "recording_data.tar.gz")
        fi
        # VOiCES_release.tar.gz (417.5 GiB) uncompressed to VOiCES_rebuilt/
        # VOiCES_devkit.tar.gz (27.5 GiB) uncompressed to VOiCES_devkit/
        # VOiCES_competition.tar.gz (19.5 GiB) uncompressed to VOiCES_Box_unzip/
        # recording_data.tar.gz (56 MiB) uncompressed to distances.csv and quality_metrics.csv
        for x in "${lst[@]}"; do
            aws s3 cp --no-sign-request s3://lab41openaudiocorpus/${x} "${VOICES}/"
            tar -zxf "${VOICES}"/${x} -C "${VOICES}"
        done
    else
        log "stage 1: ${VOICES} already exists. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"

    if $use_devkit_subsets; then
        root="${VOICES}"/VOiCES_devkit
    else
        root="${VOICES}"/VOiCES_rebuilt
    fi
    mkdir -p data/local
    #================================================================
    # Single-channel distant-16k and source-16k
    #================================================================
    awk -F ',' 'NR>1 {if(NF!=3) {exit 1;} print($2" "$3)}' \
        "${root}"/references/filename_transcripts |
        awk -F '-' '{if (substr($6,1,2)!="sp") {exit 1} sid=substr($6,3,length($6)); print(sid"_"$0)}' \
        > data/local/voices_text_distant
    awk '{
        n=split($1, uids, "-"); out=uids[1];
        for (i=2; i<=n; i++) {
            if (i==5) {out=out"-src"}
            else if (i<=3 || (i >=6 && i <= 8) || i > 12) {out=out"-"uids[i]}
        }
        $1=""; print(out" "$0)
    }' data/local/voices_text_distant | sort | uniq > data/local/voices_text_source
    for cond in "distant-16k/speech" "source-16k"; do
        if [ $cond = "distant-16k/speech" ]; then
            suffix="_distant"
        else
            suffix="_source"
        fi
        mkdir -p data/${train_set}${suffix}_tmp data/${train_set}${suffix} data/${valid_set}${suffix} data/${test_set}${suffix}
        # 12800 samples (devkit) or 661248 samples (full)
        find "${root}"/${cond}/train -iname '*.wav' | \
            awk -F '/' '{a=$NF; n=split(a, ret, ".wav"); print(ret[1]" "$0)}' | sort \
            > data/${train_set}${suffix}_tmp/wav.scp
        # 6400 samples (devkit) or 337920 samples (full)
        find "${root}"/${cond}/test -iname '*.wav' | \
            awk -F '/' '{a=$NF; n=split(a, ret, ".wav"); print(ret[1]" "$0)}' | sort \
            > data/${test_set}${suffix}/wav.scp

        if [ $cond = "distant-16k/speech" ]; then
            # e.g., Lab41-SRI-VOiCES-rm1-babb-sp0198-ch000209-sg0010-mc01-stu-clo-dg030
            awk '{print $1}' data/${train_set}${suffix}_tmp/wav.scp | \
                awk -F '-' '{if (substr($6,1,2)!="sp") {exit 1} sid=substr($6,3,length($6)); print($0" "sid)}' \
                > data/${train_set}${suffix}_tmp/utt2spk
            awk '{print $1}' data/${test_set}${suffix}/wav.scp | \
                awk -F '-' '{if (substr($6,1,2)!="sp") {exit 1} sid=substr($6,3,length($6)); print($0" "sid)}' \
                > data/${test_set}${suffix}/utt2spk
        else
            # e.g., Lab41-SRI-VOiCES-src-sp0032-ch021631-sg0005
            awk '{print $1}' data/${train_set}${suffix}_tmp/wav.scp | \
                awk -F '-' '{if (substr($5,1,2)!="sp") {exit 1} sid=substr($5,3,length($5)); print($0" "sid)}' \
                > data/${train_set}${suffix}_tmp/utt2spk
            awk '{print $1}' data/${test_set}${suffix}/wav.scp | \
                awk -F '-' '{if (substr($5,1,2)!="sp") {exit 1} sid=substr($5,3,length($5)); print($0" "sid)}' \
                > data/${test_set}${suffix}/utt2spk
        fi
        for x in "${train_set}${suffix}_tmp" "${test_set}${suffix}"; do
            awk 'FNR==NR {spk[$1]=$2; next} {print(spk[$1]"_"$0)}' \
                data/${x}/utt2spk data/${x}/wav.scp | sort > data/${x}/wav_new.scp
            mv data/${x}/wav_new.scp data/${x}/wav.scp
            awk '{print($2"_"$0)}' data/${x}/utt2spk | sort > data/${x}/utt2spk_new
            mv data/${x}/utt2spk_new data/${x}/utt2spk
        done

        utils/utt2spk_to_spk2utt.pl data/${train_set}${suffix}_tmp/utt2spk > data/${train_set}${suffix}_tmp/spk2utt
        utils/utt2spk_to_spk2utt.pl data/${test_set}${suffix}/utt2spk > data/${test_set}${suffix}/spk2utt

        utils/filter_scp.pl data/${train_set}${suffix}_tmp/wav.scp data/local/voices_text${suffix} | sort > data/${train_set}${suffix}_tmp/text
        utils/filter_scp.pl data/${test_set}${suffix}/wav.scp data/local/voices_text${suffix} | sort > data/${test_set}${suffix}/text

        # split ${train_set}${suffix}_tmp set to ${train_set} and ${valid_set}
        head -n 10 data/${train_set}${suffix}_tmp/spk2utt > data/${train_set}${suffix}_tmp/dev_spk2utt
        tail -n +11 data/${train_set}${suffix}_tmp/spk2utt > data/${train_set}${suffix}_tmp/train_spk2utt
        utils/subset_data_dir.sh --spk-list data/${train_set}${suffix}_tmp/dev_spk2utt data/${train_set}${suffix}_tmp data/${valid_set}${suffix}
        utils/subset_data_dir.sh --spk-list data/${train_set}${suffix}_tmp/train_spk2utt data/${train_set}${suffix}_tmp data/${train_set}${suffix}
        rm -r data/${train_set}${suffix}_tmp

        utils/validate_data_dir.sh --no-feats data/${train_set}${suffix}
        utils/validate_data_dir.sh --no-feats data/${valid_set}${suffix}
        utils/validate_data_dir.sh --no-feats data/${test_set}${suffix}
    done

    #================================================================
    # Multi-channel distant-16k
    #================================================================
    for x in "${train_set}" "${valid_set}" "${test_set}"; do
        python local/prepare_multich_datadir.py data/${x}_distant --outdir data/${x}_${nch}ch
        utils/utt2spk_to_spk2utt.pl data/${x}_${nch}ch/utt2spk > data/${x}_${nch}ch/spk2utt
        utils/validate_data_dir.sh --no-feats data/${x}_${nch}ch
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Combine all training and development sets"
    utils/combine_data.sh data/${train_set} data/${train_set}_distant data/${train_set}_source
    utils/combine_data.sh data/${valid_set} data/${valid_set}_distant data/${valid_set}_source
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # use external text data as in Librispeech
    if [ ! -e data/local/other_text/librispeech-lm-norm.txt.gz ]; then
        log "stage 4: prepare external text data from http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/other_text/
    fi
    if [ ! -e data/local/other_text/text ]; then
        # provide utterance id to each texts
        # e.g., librispeech_lng_00003686 A BANK CHECK
        zcat data/local/other_text/librispeech-lm-norm.txt.gz | \
            awk '{ printf("librispeech_lng_%08d %s\n",NR,$0) } ' > data/local/other_text/text
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
