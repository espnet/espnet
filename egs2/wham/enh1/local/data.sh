#!/bin/bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

. ./db.sh

wsj_full_wav=$PWD/data/wsj0/wsj0_wav
wham_wav=$PWD/data/wham/2speakers
wham_scripts=$PWD/data/wham


other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt
min_or_max=max
sample_rate=16k


. utils/parse_options.sh


if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi
if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi

train_set="tr_"${min_or_max}_${sample_rate}
train_dev="cv_"${min_or_max}_${sample_rate}
recog_set="tt_"${min_or_max}_${sample_rate}



### This part is for WHAM!
### Download mixture scripts and create mixtures for 2 speakers
local/wham_create_mixture.sh --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
   --wsj0_2mix /mnt/lustre/sjtu/home/cdl54/workspace/asr/develop/espnet/egs2/wsj0_2mix/asr1/data/wsj0_mix/2speakers \
   --wham_noise /mnt/lustre/sjtu/shared/data/asr/rawdata/wham_noise \
   ${wham_scripts} ${WSJ0} ${wsj_full_wav} \
   ${wham_wav} || exit 1;

# The following datasets will be created:
# {tr,cv,tt}_mix_{both,clean,single}_${min_or_max}_${sample_rate}
#
# Note:
#   - `both`: a mixture of speech1, speech2 and noise (for speech separation)
#   - `clean`: a mixture of speech1 and speech2 (for speech separation)
#   - `single`: a mixture of speech1 and noise (for speech enhancement)
local/wham_data_prep.sh --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
    ${wham_scripts}/wham_scripts ${wham_wav} ${wsj_full_wav} || exit 1;

echo 'data prep done'
exit 0;

### create .scp file for reference audio
for folder in ${train_set} ${train_dev} ${recog_set};
do
    sed -e 's/\/mix\//\/s1\//g' ./data/$folder/wav.scp > ./data/$folder/spk1.scp
    sed -e 's/\/mix\//\/s2\//g' ./data/$folder/wav.scp > ./data/$folder/spk2.scp
done


### Also need wsj corpus to prepare language information
### This is from Kaldi WSJ recipe
log "local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?"
local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
log "local/wsj_format_data.sh"
local/wsj_format_data.sh
log "mkdir -p data/wsj"
mkdir -p data/wsj
log "mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj"
mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj



log "Prepare text from lng_modl dir: ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z -> ${other_text}"
mkdir -p "$(dirname ${other_text})"

# NOTE(kamo): Give utterance id to each texts.
zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
    grep -v "<" | tr "[:lower:]" "[:upper:]" | \
    awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}



log "Create non linguistic symbols: ${nlsyms}"
cut -f 2- data/wsj/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
cat ${nlsyms}
