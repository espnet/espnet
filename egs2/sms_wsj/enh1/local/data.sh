#!/usr/bin/env bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Note: This script is based on the original repository of SMS-WSJ:
#  https://github.com/fgnt/sms_wsj
# and the sms_wsj recipe in Asteroid:
#  https://github.com/mpariente/asteroid/blob/master/egs/sms_wsj/CaCGMM/local/prepare_data.sh
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh


nj=16
sample_rate=8k
num_spk=2   # one of (2, 3, 4)

# True to use reverberated sources in spk?.scp
# False to use original source signals (padded) in spk?.scp
use_reverb_reference=true
download_rir=true

. utils/parse_options.sh

. ./path.sh


sms_wsj_wav=$PWD/data/sms_wsj
wsj_zeromean_wav=${sms_wsj_wav}/wsj_${sample_rate}_zeromean
local_dir=$PWD/local
sms_wsj_scripts=$PWD/local/sms_wsj
other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt

if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi
if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi


if [[ ! -d ${sms_wsj_scripts} ]]; then
    log "Cloning and installing SMS-WSJ repository"
    git clone https://github.com/fgnt/sms_wsj.git ${sms_wsj_scripts}
    (
        cd ${sms_wsj_scripts}
        git checkout f0cc4402a7111cd603c617a21170b0e0ddb90d48
        git apply ${local_dir}/write_additional_data.patch
    )
    # Note: MPI pre-installation is required here.
    python -m pip install -e ${sms_wsj_scripts}
    if ! ${download_rir}; then
        git clone https://github.com/boeddeker/rir-generator.git ${sms_wsj_scripts}/reverb/rirgen_rep
        python -m pip install -e ${sms_wsj_scripts}/reverb/rirgen_rep/python/
    fi
fi

local/create_database.sh \
    --nj ${nj} \
    --sample-rate ${sample_rate} \
    --num-spk ${num_spk} \
    --download-rir ${download_rir} \
    ${WSJ0} ${WSJ1} ${wsj_zeromean_wav} ${sms_wsj_wav} || exit 1;

# The following datasets will be created using the default configuration:
#  - train_si284: 33561 samples (87:22:26)
#  - cv_dev93:    982 samples   (02:31:51)
#  - test_eval92: 1332 samples  (03:21:23)
# The data files are generated based on the sms_wsj.json file,
# and the utterance ids are slightly modified based those in sms_wsj.json.
python local/sms_wsj_data_prep.py \
    --num-spk ${num_spk} \
    --sample-rate ${sample_rate} \
    --use-reverb-reference ${use_reverb_reference} \
    --dist-dir data \
    ${sms_wsj_wav}/sms_wsj.json || exit 1;

for subset in train_si284 cv_dev93 test_eval92; do
    files="noise1.scp utt2dur utt2spk wav.scp"
    for i in $(seq ${num_spk}); do
        files+=" dereverb${i}.scp rir${i}.scp spk${i}.scp text_spk${i}"
    done
    for f in ${files}; do
        mv data/${subset}/${f} data/${subset}/.${f}
        sort data/${subset}/.${f} > data/${subset}/${f}
        rm data/${subset}/.${f}
    done
    utils/utt2spk_to_spk2utt.pl data/${subset}/utt2spk > data/${subset}/spk2utt
    awk '{print($1, "reverb_6channels")}' data/${subset}/utt2spk > data/${subset}/utt2category
    utils/validate_data_dir.sh --no-feats --no-text data/${subset}
done

### Also need wsj corpus to prepare language information
### This is from Kaldi WSJ recipe (may take ~40 minutes)
log "local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?"
local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
log "mkdir -p data/wsj"
mkdir -p data/wsj
log "local/wsj_format_data.sh"
local/wsj_format_data.sh --data_dir data/wsj
mv data/local data/wsj
# only for multi-condition training in ASR
ln -s wsj/train_si284 data/wsj_train_si284
for i in $(seq ${num_spk}); do
    ln -s text data/wsj_train_si284/text_spk$i
done
awk '{print($1, "single_channel")}' data/wsj_train_si284/utt2spk > data/wsj_train_si284/utt2category


log "Prepare text from lng_modl dir: ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z -> ${other_text}"
mkdir -p "$(dirname ${other_text})"

# NOTE(kamo): Give utterance id to each texts.
zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
    grep -v "<" | tr "[:lower:]" "[:upper:]" | \
    awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}


log "Create non linguistic symbols: ${nlsyms}"
cut -f 2- data/wsj/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
cat ${nlsyms}
