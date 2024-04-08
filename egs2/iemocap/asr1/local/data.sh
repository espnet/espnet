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
stop_stage=100

lowercase=false
# Convert into lowercase if "true".
remove_punctuation=false
# Remove punctuation (except apostrophes) if "true".
# Note that punctuation normalization will be performed in the "false" case.
remove_tag=false
# Remove [TAGS] (e.g.[LAUGHTER]) if "true".
remove_emo=
# Remove the utterances with the specified emotional labels
# emotional labels: ang (anger), hap (happiness), exc (excitement), sad (sadness),
# fru (frustration), fea (fear), sur (surprise), neu (neutral), and xxx (other)
convert_to_sentiment=false
# for sentiment (positive, negative and neutral) analysis
# mapping from emotion to sentiment is as follows:
# Positive: hap, exc, sur
# Negative: ang, sad, fru, fea
# Neutral: neu

#data
datadir=/ocean/projects/cis210027p/shared/corpora/IEMOCAP_full_release
# IEMOCAP_full_release
#  |_ README.txt
#  |_ Documentation/
#  |_ Session{1-5}/
#      |_ sentences/wav/ ...<wav files for each utterance>
#      |_ dialog/
#          |_ transcriptions/ ...<transcription files>
#          |_ EmoEvaluation/ ...<emotion annotation files>
# In this recipe
# Sessions 1-3 & 4F (Ses01, SeS02, Ses03,and Ses04F) are used for training (6871 utterances),
# Session 4M (Ses04M) is used for validation (998 utterances), and
# Session 5 (Ses05) is used for evaluation (2170 utterances).
# Download data from here:
# https://sail.usc.edu/iemocap/

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: IEMOCAP Data Preparation"
    # This process may take a few minutes for the first run
    # Remove "data/${tmp}/tmp.done" if you want to start all over again
    if [ -n "${remove_emo}" ]; then
        log "Remove ${remove_emo} from emotional labels"
        tmp="tmp/remove_emo"
    else
        log "Use all 9 emotional labels"
        tmp=tmp
    fi
    if [ ! -e data/${tmp}/tmp.done ];then
        mkdir -p data/{train,valid,test}
        mkdir -p data/${tmp}
        echo -n "" > data/${tmp}/wav.scp; echo -n "" > data/${tmp}/utt2spk; echo -n "" > data/${tmp}/text
        for n in {1..5}; do
            for file in "${datadir}"/Session"${n}"/sentences/wav/*/*.wav; do
                utt_id=$(basename ${file} .wav)
                ses_id=$(echo "${utt_id}" | sed "s/_[FM][0-9]\{3\}//g")
                words=$(grep ${utt_id} ${datadir}/Session${n}/dialog/transcriptions/${ses_id}.txt \
                        | sed "s/^.*\]:\s\(.*\)$/\1/g")
                emo=$(grep ${utt_id} ${datadir}/Session${n}/dialog/EmoEvaluation/${ses_id}.txt \
                        | sed "s/^.*\t${utt_id}\t\([a-z]\{3\}\)\t.*$/\1/g")
                if ! eval "echo ${remove_emo} | grep -q ${emo}" ; then
                    # for sentiment analysis
                    if [ ${convert_to_sentiment} = "true" ]; then
                        words2=$(echo "$words" | perl local/prepare_sentiment.pl)
                        if [ ${emo} = "hap" ] || [ ${emo} = "exc" ] || [ ${emo} = "sur" ]; then
                            echo "${utt_id} Positive ${words2}" >> data/${tmp}/text
                            echo "${utt_id} ${file}" >> data/${tmp}/wav.scp
                            echo "${utt_id} ${utt_id}" >> data/${tmp}/utt2spk
                        elif [ ${emo} = "ang" ] || [ ${emo} = "sad" ] || [ ${emo} = "fru" ] || [ ${emo} = "fea" ]; then
                            echo "${utt_id} Negative ${words2}" >> data/${tmp}/text
                            echo "${utt_id} ${file}" >> data/${tmp}/wav.scp
                            echo "${utt_id} ${utt_id}" >> data/${tmp}/utt2spk
                        elif [ ${emo} = "neu" ];then
                            echo "${utt_id} Neutral ${words2}" >> data/${tmp}/text
                            echo "${utt_id} ${file}" >> data/${tmp}/wav.scp
                            echo "${utt_id} ${utt_id}" >> data/${tmp}/utt2spk
                        fi
                    else
                        echo "${utt_id} <${emo}> ${words}" >> data/${tmp}/text
                        echo "${utt_id} ${file}" >> data/${tmp}/wav.scp
                        echo "${utt_id} ${utt_id}" >> data/${tmp}/utt2spk
                    fi
                fi
            done
        done
        dos2unix data/${tmp}/wav.scp; dos2unix data/${tmp}/utt2spk; dos2unix data/${tmp}/text
        utils/utt2spk_to_spk2utt.pl data/${tmp}/utt2spk > "data/${tmp}/spk2utt"
        touch data/${tmp}/tmp.done
    fi
    for file in wav.scp utt2spk spk2utt text; do
        grep -e "Ses01" -e "Ses02" -e "Ses03" -e "Ses04F" data/${tmp}/${file} > data/train/${file}
        grep "Ses04M" data/${tmp}/${file} > data/valid/${file}
        grep "Ses05" data/${tmp}/${file} > data/test/${file}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ ${convert_to_sentiment} != "true" ]; then
        log "stage 2: IEMOCAP Transcript Conversion"
        mkdir -p data/{train,valid,test}/{original,tmp}/
        for dset in train valid test; do
            cp data/${dset}/text -t data/${dset}/original/
            if ${lowercase}; then
                log "lowercase ${dset}"
                perl local/lowercase.pl < data/${dset}/text > data/${dset}/tmp/text
                cp data/${dset}/tmp/text data/${dset}/text
            fi
            if ${remove_punctuation}; then
                log "remove_punctuation ${dset}"
                perl local/remove_punctuation.pl < data/${dset}/text > data/${dset}/tmp/text
                cp data/${dset}/tmp/text data/${dset}/text
            fi
            if ${remove_tag}; then
                log "remove_tag ${dset}"
                perl local/remove_tag.pl < data/${dset}/text > data/${dset}/tmp/text
                cp data/${dset}/tmp/text data/${dset}/text
            fi
            #Remove extra space and normalize punctuation
            perl local/normalize_punctuation.pl < data/${dset}/text > data/${dset}/tmp/text
            cp data/${dset}/tmp/text data/${dset}/text
        done
    fi
    for dset in test valid train; do
        utils/validate_data_dir.sh --no-feats data/${dset} || exit 1
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
