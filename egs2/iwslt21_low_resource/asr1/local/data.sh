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

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation"
    local/prepare_babel_data.py ${BABEL_202}
    local/prepare_alffa_data.py ${ALFFA}/data_broadcastnews_sw
    local/prepare_gamayun_data.py ${GAMAYUN}/gamayun-swahili
    local/prepare_iwslt_data.py ${IWSLT21LR}/IWSLT-lowresource
    local/prepare_iwslt_data.py --raw-transcriptions ${IWSLT21LR}/IWSLT-lowresource

    for d in data/*; do
        utils/fix_data_dir.sh ${d}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: combine datasets"
    utils/combine_data.sh --extra_files utt2num_frames data/train_babel_alffa_gamayun_iwslt \
        data/train_babel \
        data/train_alffa \
        data/train_gamayun \
        data/train_iwslt_swa \
        data/train_iwslt_swc
    utils/combine_data.sh --extra_files utt2num_frames data/valid_alffa_iwslt \
        data/test_alffa \
        data/valid_iwslt_swa \
        data/valid_iwslt_swc
    utils/combine_data.sh --extra_files utt2num_frames data/test_iwslt \
        data/test_iwslt_swa \
        data/test_iwslt_swc
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: prepare external text data"

    mkdir -p data/local/other_text
    sed -e 's/<s> //g' -e 's/ <\/s>//g' ${ALFFA}/data_broadcastnews_sw/LM/01-CLN4-TRN.txt | \
      iconv -f "UTF-8" -t "UTF-8//IGNORE" | \
      perl -p -e '$_="" if length($_) > 600' | \
      awk '{ printf("alffa_lng_%07d %s\n",NR,$0) } ' > data/local/other_text/text
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: download pre-trained ASR model"

    mkdir -p downloads
    wget "https://zenodo.org/record/4541727/files/asr_train_asr_conformer_raw_ru_bpe100_valid.acc.ave.zip?download=1" \
      -O downloads/ru_open_stt.zip

    unzip downloads/ru_open_stt.zip \
      exp/asr_train_asr_conformer_raw_ru_bpe100/valid.acc.ave_10best.pth \
      -d downloads/ -f

    mv downloads/exp/asr_train_asr_conformer_raw_ru_bpe100/valid.acc.ave_10best.pth \
      downloads/ru_open_stt_conformer.pth

    wget "https://zenodo.org/record/5227612/files/swahili-asr-resources.tar.xz?download=1" \
       -O downloads/swahili-asr-resources.tar.xz

    tar xf downloads/swahili-asr-resources.tar.xz -C downloads

    rm -rf downloads/{exp,ru_open_stt.zip,swahili-asr-resources.tar.xz}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
