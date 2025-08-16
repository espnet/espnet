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

dummy_data=false
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
datadir=/ocean/projects/cis210027p/shared/corpora/msp_podcast_v1.12
# msp_podcast
#  |_ readme.txt
#  |_ Partitions.txt
#  |_ Speaker_ids.txt
#  |_ ForceAligned/
#  |_ Labels/
#  |_ Transcripts/
#  |_ Audios/
#      |_ ...<wav files for each utterance>
# In this recipe
# Training/Development/Testing sets followed the setting in Partitions.txt
# Training set includes 112712 utterances
# Development set includes 31961 utterances
# Test1 set includes 44395 utterances
# Test2 set includes 14868 utterances
# Test3 set includes 3200 utterances
# Download data from here:
# https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html

log "$0 $*"
. utils/parse_options.sh || exit 1

. ./path.sh
. ./cmd.sh

if [ "$dummy_data" = true ]; then
    log "Stage 1 (dummy): Generating 10 random dummy samples for unit tests"

    # Cleanup old
    rm -rf data/{train,valid,test1,test2,tmp}

    # Create dirs & empty metadata files
    for split in train valid test1 test2 test tmp; do
        mkdir -p data/$split
        for f in wav.scp utt2spk text utt2emo split_set; do
            : > data/$split/$f
        done
    done

    emotions=(A S H U F D C N O X)
    # Generate 10 utts, assign splits:
    #  1–10 → Train, 11–12 → Development, 13 → Test1, 14 → Test2, 15 → Test
    for i in $(seq 1 15); do
        utt=$(printf "dummy_%02d" $i)
        if   [ $i -le 10 ]; then split_set="Train"
        elif [ $i -le 12 ]; then split_set="Development"
        elif [ $i -eq 13 ]; then split_set="Test1"
        elif [ $i -eq 14 ]; then split_set="Test2"
        else                    split_set="Test"
        fi
        text="This is dummy text for ${utt}"
        spk=$(printf "spk%02d" $(( (i - 1) % 3 + 1 )))
        emo=${emotions[$(( (i-1) % ${#emotions[@]} ))]}

        # Create 1-second silent wav at 16kHz mono
        wavpath="data/tmp/${utt}.wav"
        mkdir -p "$(dirname "$wavpath")"
        if command -v sox >/dev/null 2>&1; then
            sox -n -r 16000 -c 1 "$wavpath" trim 0 1 >/dev/null
        elif command -v ffmpeg >/dev/null 2>&1; then
            ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 1 "$wavpath" >/dev/null 2>&1
        elif command -v python3 >/dev/null 2>&1; then
            python3 - <<EOF
import wave
f = wave.open("${wavpath}", "w")
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(16000)
f.writeframes(b'\x00\x00' * 16000)
f.close()
EOF
        else
            log "Error: please install sox, ffmpeg, or have python3 available to generate dummy wav files"
            exit 1
        fi

        echo "${utt} ${text}"            >> data/tmp/text
        echo "${utt} ${wavpath}"         >> data/tmp/wav.scp
        echo "${utt} ${spk}"             >> data/tmp/utt2spk
        echo "${utt} ${emo}"             >> data/tmp/utt2emo
        echo "${utt} ${split_set}"       >> data/tmp/split_set
    done

    touch data/tmp/tmp.done

    # Split into train/valid/test*
    for file in wav.scp text utt2spk utt2emo; do
        while read -r line; do
            u=$(echo "$line" | awk '{print $1}')
            s=$(grep "^${u} " data/tmp/split_set | awk '{print $2}')
            case $s in
                Train)       dst="train" ;;
                Development) dst="valid" ;;
                Test1)       dst="test1" ;;
                Test2)       dst="test2" ;;
                Test)        dst="test" ;;
                *) log "Unknown split ${s} for ${u}" && exit 1 ;;
            esac
            echo "$line" >> data/${dst}/${file}
        done < data/tmp/${file}
    done

    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
    utils/utt2spk_to_spk2utt.pl data/valid/utt2spk > data/valid/spk2utt
    utils/utt2spk_to_spk2utt.pl data/test1/utt2spk > data/test1/spk2utt
    utils/utt2spk_to_spk2utt.pl data/test2/utt2spk > data/test2/spk2utt
    utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

    # for dset in train valid test1 test2 test; do
    #     utils/validate_data_dir.sh --no-feats --no-spk-sort data/${dset} || exit 1
    # done

    log "Dummy data prep complete: data/{train,valid,test1,test2,test}"
    exit 0
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: MSP_Podcast Data Preparation"
    # This process may take a few minutes for the first run
    # Remove "data/${tmp}/tmp.done" if you want to start all over again
    if [ -n "${remove_emo}" ]; then
        log "Remove ${remove_emo} from emotional labels"
        tmp="tmp/remove_emo"
    else
        log "Use all 10 emotional labels"
        tmp=tmp
    fi
    if [ ! -e data/${tmp}/tmp.done ];then
        mkdir -p data/{train,valid,test1,test2}
        mkdir -p data/${tmp}
        echo -n "" > data/${tmp}/wav.scp; echo -n "" > data/${tmp}/utt2spk; echo -n "" > data/${tmp}/text
        tail -n +2 "${datadir}/Labels/labels_consensus.csv" | while IFS=',' read -r FileName EmoClass _ _ _ SpkrID Gender Split_Set
        do
            utt_id=$(basename ${FileName} .wav)
            spk_id=${SpkrID}
            gender=${Gender}
            emo_cate=${EmoClass}
            split_set=${Split_Set}
            file="${datadir}/Audios/${FileName}"
            words=$(cat "${datadir}"/Transcripts/${utt_id}.txt | sed "s/^.*\]:\s\(.*\)$/\1/g")
            if ! eval "echo ${remove_emo} | grep -q ${emo_cate}" ; then
                # for sentiment analysis
                if [ ${convert_to_sentiment} = "true" ]; then
                    words2=$(echo "$words" | perl local/prepare_sentiment.pl)
                    if [ ${emo_cate} = "hap" ] || [ ${emo_cate} = "exc" ] || [ ${emo_cate} = "sur" ]; then
                        echo "${utt_id} Positive ${words2}" >> data/${tmp}/text
                        echo "${utt_id} ${file}" >> data/${tmp}/wav.scp
                        echo "${utt_id} ${utt_id}" >> data/${tmp}/utt2spk
                    elif [ ${emo_cate} = "ang" ] || [ ${emo_cate} = "sad" ] || [ ${emo_cate} = "fru" ] || [ ${emo_cate} = "fea" ]; then
                        echo "${utt_id} Negative ${words2}" >> data/${tmp}/text
                        echo "${utt_id} ${file}" >> data/${tmp}/wav.scp
                        echo "${utt_id} ${utt_id}" >> data/${tmp}/utt2spk
                    elif [ ${emo_cate} = "neu" ];then
                        echo "${utt_id} Neutral ${words2}" >> data/${tmp}/text
                        echo "${utt_id} ${file}" >> data/${tmp}/wav.scp
                        echo "${utt_id} ${utt_id}" >> data/${tmp}/utt2spk
                    fi
                else
                    echo "${utt_id} ${words}" >> data/${tmp}/text
                    echo "${utt_id} ${file}" >> data/${tmp}/wav.scp
                    echo "${utt_id} ${spk_id}" >> data/${tmp}/utt2spk
                    echo "${utt_id} ${gender}" >> data/${tmp}/utt2gen
                    echo "${utt_id} ${split_set}" >> data/${tmp}/split_set
                    echo "${utt_id} ${emo_cate}" >> data/${tmp}/utt2emo
                fi
            fi
        done
        dos2unix data/${tmp}/wav.scp; dos2unix data/${tmp}/text; dos2unix data/${tmp}/split_set; dos2unix data/${tmp}/utt2emo; dos2unix data/${tmp}/utt2spk; dos2unix data/${tmp}/utt2gen
        # utils/utt2spk_to_spk2utt.pl data/${tmp}/utt2spk > "data/${tmp}/spk2utt"
        touch data/${tmp}/tmp.done
    fi

    for file in wav.scp text utt2spk utt2emo; do
        while read -r line; do
            utt_id=$(echo "$line" | awk '{print $1}')
            split_set=$(grep "^${utt_id} " "data/${tmp}/split_set" | awk '{print $2}')

            # If utt_id is found in utt2split and split_set is known
            if [ "$split_set" = "Train" ]; then
                echo "$line" >> data/train/${file}
            elif [ "$split_set" = "Development" ]; then
                echo "$line" >> data/valid/${file}
            elif [ "$split_set" = "Test1" ]; then
                echo "$line" >> data/test1/${file}
            elif [ "$split_set" = "Test2" ]; then
                echo "$line" >> data/test2/${file}
            else
                log "Unknown split set: $split_set for utt_id: $utt_id"
            fi
        done < "data/${tmp}/${file}"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ ${convert_to_sentiment} != "true" ]; then
        log "stage 2: MSP-Pocast Transcript Conversion"
        utils/combine_data.sh --extra-files utt2emo --skip_fix true data/test data/test1 data/test2
        mkdir -p data/{train,valid,test1,test2,test}/{original,tmp}/
        for dset in train valid test1 test2 test; do
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

    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
    utils/utt2spk_to_spk2utt.pl data/valid/utt2spk > data/valid/spk2utt
    utils/utt2spk_to_spk2utt.pl data/test1/utt2spk > data/test1/spk2utt
    utils/utt2spk_to_spk2utt.pl data/test2/utt2spk > data/test2/spk2utt
    utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

    # for dset in train valid test1 test2 test; do
    #     utils/validate_data_dir.sh --no-feats --no-spk-sort data/${dset} || exit 1
    # done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
