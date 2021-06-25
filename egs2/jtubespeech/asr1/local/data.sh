#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=10000

thre=-0.3
maxchars=80

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

data_tag="_ss0622_th${thre}"
odir=/exp/swatanabe/data/opj/single-speaker
scores=/exp/swatanabe/data/opj/single-speaker/segments/segments_20210531_ctcscore.txt
scores=/expscratch/swatanabe/202007espnet/espnet_v2/egs2/jtubespeech/asr1/100G_ctcseg_0622/segments.txt

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "prepare the basic data directories"
    mkdir -p data/local/data${data_tag}
    find ${odir}/wav16k/ -name '*.wav' | sort > data/local/data${data_tag}/wav.flist
    awk -F '/' '{print $NF}' data/local/data${data_tag}/wav.flist | sed -e "s/\.wav//" > data/local/data${data_tag}/id.list
    paste -d ' ' data/local/data${data_tag}/id.list data/local/data${data_tag}/wav.flist > data/local/data${data_tag}/wav.scp

    top_scores=data/local/data${data_tag}/segments_scores
    log "save more than ${thre} score file to ${top_scores}"
    awk -v t="${thre}" '$5 > t' ${scores} > ${top_scores}
    nutt=$(wc -l ${top_scores} | awk '{print $1}')
    log "we will use top ${nutt} utterances"

    log "get segments from the score file"
    awk '{print $1 " " $2 " " $3 " " $4}' ${top_scores} | sort > data/local/data${data_tag}/segments

    log "make utt2spk and spk2utt from data/local/data${data_tag}/segments"
    paste -d ' ' <(awk '{print $1}' data/local/data${data_tag}/segments) <(awk '{print $2}' data/local/data${data_tag}/segments) > data/local/data${data_tag}/utt2spk
    utils/utt2spk_to_spk2utt.pl data/local/data${data_tag}/utt2spk > data/local/data${data_tag}/spk2utt

    log "convert unicode space to the ascii space with \`perl -CSDA -plE 's/\s/ /g'\`"
    paste -d ' ' <(awk '{print $1}' ${top_scores}) <(cut -f 6- -d" " ${top_scores} | perl -CSDA -plE 's/\s/ /g') > data/local/data${data_tag}/text
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "remove too long or too short utterances"
    rm -fr data/local/data${data_tag}_top${nutt}
    remove_longshortdata.sh --maxchars ${maxchars} data/local/data${data_tag} data/local/data${data_tag}_trim
fi

#if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#    log "split the data into training and validation data by randomly picking up 5% of the recordings as a validation set"
#    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/local/data${data_tag}_trim data/train${data_tag} data/valid${data_tag}
#    utils/fix_data_dir.sh data/train${data_tag}
#    utils/fix_data_dir.sh data/valid${data_tag}
#fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "make the training data by subtracting the test set speaker list"
    awk '{print $1}' data/local/data${data_tag}_trim/spk2utt | sort > data/local/data${data_tag}_trim/all_speaker_list
    comm -3 data/local/data${data_tag}_trim/all_speaker_list local/test_speaker_list > data/local/data${data_tag}_trim/train_speaker_list
    utils/subset_data_dir.sh --spk-list data/local/data${data_tag}_trim/train_speaker_list data/local/data${data_tag}_trim data/train${data_tag}_nodev
    utils/fix_data_dir.sh data/train${data_tag}_nodev
fi

#if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#    log "make more difficult test data by using the test set speaker list and changing the threshold"
#    utils/subset_data_dir.sh --spk-list local/test_speaker_list data/local/data${data_tag}_trim data/valid${data_tag}_fixspeaker
#    utils/fix_data_dir.sh data/valid${data_tag}_fixspeaker
#fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "make an official easy_jun21 test set"
    for d_or_e in dev eval; do
	# we will add normal, hard, and hell modes in the future
	# shellcheck disable=SC2043
	for test_mode in easy_jun21; do
	    x=${d_or_e}_${test_mode}
	    if [ ! -d data/${x} ]; then
		log "get segments from local/${x}.list"
		mkdir -p data/local/data_${x}
		grep -f local/${x}.list ${scores} > data/local/data_${x}/segments_scores
		awk '{print $1 " " $2 " " $3 " " $4}' data/local/data_${x}/segments_scores | sort > data/local/data_${x}/segments

		log "make utt2spk and spk2utt from data/local/data_${x}/segments"
		paste -d ' ' <(awk '{print $1}' data/local/data_${x}/segments) <(awk '{print $2}' data/local/data_${x}/segments) > data/local/data_${x}/utt2spk
		utils/utt2spk_to_spk2utt.pl data/local/data_${x}/utt2spk > data/local/data_${x}/spk2utt

		log "convert unicode space to the ascii space with \`perl -CSDA -plE 's/\s/ /g'\`"
		paste -d ' ' <(awk '{print $1}' data/local/data_${x}/segments_scores) <(cut -f 6- -d" " data/local/data_${x}/segments_scores \
		    | perl -CSDA -plE 's/\s/ /g') | sort > data/local/data_${x}/text

		log "make a wav file list from spk2utt"
		awk '{print $1}' data/local/data_${x}/spk2utt > data/local/data_${x}/spk.list
		find ${odir}/wav16k/ -name '*.wav' | grep -f data/local/data_${x}/spk.list | sort > data/local/data_${x}/wav.flist
		awk -F '/' '{print $NF}' data/local/data_${x}/wav.flist | sed -e "s/\.wav//" > data/local/data_${x}/id.list
		paste -d ' ' data/local/data_${x}/id.list data/local/data_${x}/wav.flist | sort > data/local/data_${x}/wav.scp

		utils/copy_data_dir.sh data/local/data_${x} data/${x}
		utils/fix_data_dir.sh data/${x}
	    fi
	done
    done
    # for now we just copy dev_easy_jun21, but we will add other tasks
    log "make a validation set"
    if [ ! -d data/dev_easy_jun21 ]; then
	utils/copy_data_dir.sh data/dev_easy_jun21 data/valid
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
