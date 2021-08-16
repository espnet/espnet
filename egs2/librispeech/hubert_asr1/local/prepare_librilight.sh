#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $# -ne 1 ]; then
  echo "Usage: ./local/prepare_librilight.sh <download-data-path>"
  exit 1;
fi

data=$1

# Download libri-light for finetuning
src=${data}/librispeech_finetuning
if [ ! -d ${src}/1h ] && [ ! -d ${src}/9h ]; then
    echo "Data Download to ${data}"
    mkdir -p ${data}
    wget https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz -P ${data}
    tar vxfz ${data}/librispeech_finetuning.tgz -C ${data}
else
    log "${data}/librispeech_finetuning is already existing. Skip data downloading"
fi

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
    done | sort > ${data_new_path}/wav.scp
    
    paste -d' ' <(awk '{print $1}' ${data_new_path}/wav.scp) \
          <(awk '{print $1}' ${data_new_path}/wav.scp | cut -d'-' -f1) \
          > ${data_new_path}/utt2spk
    ./utils/utt2spk_to_spk2utt.pl ${data_new_path}/utt2spk > ${data_new_path}/spk2utt
    cat `find -L ${data_part} -name "*.trans.txt"` | sort > ${data_new_path}/text
done

./utils/combine_data.sh \
  data/train_10h data/train_1h_{0..5}_{clean,other} data/train_9h_{clean,other}
