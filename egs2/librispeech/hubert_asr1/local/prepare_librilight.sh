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
data_url=https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
train_set="train_10h"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRILIGHT}" ]; then
    log "Fill the value of 'LIBRILIGHT' of db.sh"
    exit 1
fi


src=${LIBRILIGHT}/librispeech_finetuning
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data Download to ${LIBRILIGHT}"
    if [ ! -d ${src}/1h ] && [ ! -d ${src}/9h ]; then
	mkdir -p ${LIBRILIGHT}
	wget ${data_url} -P ${LIBRILIGHT}
	tar vxfz ${LIBRILIGHT}/librispeech_finetuning.tgz -C ${LIBRILIGHT}
    else
	log "${LIBRILIGHT}/librispeech_finetuning is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    for part in 1h/{0..5}/{clean,other} 9h/{clean,other}; do
	dataname=$(echo ${part} | sed 's/\//_/g')
	
	data_part=$(./utils/make_absolute.sh ${src}/${part})
	data_new_path=data/train_${dataname}
	mkdir -p ${data_new_path}
	files=( `find -L ${data_part}/ -name "*.flac"` )
	
	for f in ${files[@]}; do
	    filename=`basename $f`
	    filename=${filename%%.flac}
	    echo "${filename} flac -c -d -s ${f} |" 
	done | sort | uniq > ${data_new_path}/wav.scp
	
	paste -d' ' <(awk '{print $1}' ${data_new_path}/wav.scp) \
              <(awk '{print $1}' ${data_new_path}/wav.scp | cut -d'-' -f1) \
              > ${data_new_path}/utt2spk
	./utils/utt2spk_to_spk2utt.pl ${data_new_path}/utt2spk > ${data_new_path}/spk2utt
	cat `find -L ${data_part} -name "*.trans.txt"` | sort > ${data_new_path}/text
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: combine 10hr training sets"
    ./utils/combine_data.sh \
	data/${train_set} data/train_1h_{0..5}_{clean,other} data/train_9h_{clean,other}
fi
