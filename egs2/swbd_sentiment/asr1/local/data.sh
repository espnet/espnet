#!/bin/bash
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
stop_stage=4

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${SWBD}" ]; then
    log "Fill the value of 'SWBD' of db.sh"
    exit 1
fi


# we assume that LDC97S62 & speech_sentiment_annotations are placed under SWBD
swbd1_dir=${SWBD}/LDC97S62
swbd_sentiment=${SWBD}/speech_sentiment_annotations/data/sentiment_labels.tsv

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log " Data Preparation"
    local/swbd1_data_download.sh ${swbd1_dir}
    local/swbd1_prepare_dict.sh
    local/swbd1_data_prep.sh ${swbd1_dir}
    # upsample audio from 8k to 16k to make a recipe consistent with others
    sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/train/wav.scp
    utils/fix_data_dir.sh data/train
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log " Data Formatting"
     # remove ._ . _1 symbols from text  
     cp data/train/text data/train/text.backup
     sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/train/text
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log " Concatenate Sentiment with Transcription"
    # Concatenate sentiment (Positive, Negative, Neutral) with transcription. 
    # Using sentiment annotation reconciliation strategy based on majority voting as in
    # https://catalog.ldc.upenn.edu/docs/LDC2020T14/LREC_2020_Switchboard_Senti.pdf
    # This stage may take a while
    mkdir -p data/local/tmp/
    mv -f data/train/* data/local/tmp/.
    mkdir -p data/dev/
    mkdir -p data/test/
    python3 local/prepare_sentiment.py \
        --train_dir data/train/ \
        --dev_dir data/dev/ \
        --test_dir data/test/ \
        --sentiment_file ${swbd_sentiment} \
        --text_file data/local/tmp/text \
        --wavscp_file data/local/tmp/wav.scp
    for dir in train dev test; do
    utils/utt2spk_to_spk2utt.pl data/$dir/utt2spk > data/$dir/spk2utt
    utils/fix_data_dir.sh data/$dir
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
