#!/bin/bash

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
# inclusive, was 100
SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. utils/parse_options.sh

mkdir -p ${ACCENTED_FR}
if [ -z "${ACCENTED_FR}" ]; then
    log "Fill the value of 'ACCENTED_FR' of db.sh"
    exit 1
fi

log "data preparation started"



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "sub-stage 0: Download Data to downloads"

    (
    cd ${ACCENTED_FR}
    wget https://www.openslr.org/resources/57/African_Accented_French.tar.gz
    tar -xvf African_Accented_French.tar.gz
    rm -r African_Accented_French.tar.gz
    )

fi

# some samples are missing (less than 0.1%), we use the following scripts to clean the datasets
python3 local/normalize_test.py --path_test "${ACCENTED_FR}/African_Accented_French/transcripts/test/ca16/"
python3 local/remove_missing.py --folder "downloads/African_Accented_French/" --train "transcripts/train/" --devtest "transcripts/devtest/"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "sub-stage 1: Preparing Data for train"

    # train set ca16_conv:
    FILE=${ACCENTED_FR}/African_Accented_French/transcripts/train/ca16_conv/new_transcripts.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '_' -f 3 "$FILE" > aux5
    cut -d ' ' -f 1 "$FILE" > uttid1
    cut -c -32 uttid1 > uttid2
    cut -d ' ' -f 2- "$FILE" > aux10

    paste -d ' ' uttid2 aux10 > auxtext1

    # take everything in aux $0, add that before to the $0 -> aux2
    awk '{print "downloads/African_Accented_French/speech/train/ca16/"$0}' aux5 > aux6

    # aux2/uttid -> aux3
    paste -d "/"  aux6 uttid2 > aux7
    awk '{print $0".wav"}' aux7 > aux8
    paste  -d " " uttid2 aux8 > auxwav1

    # identity function
    paste  -d " " uttid2 uttid2  > auxutt1

    # train set ca16_conv:
    FILE=${ACCENTED_FR}/African_Accented_French/transcripts/train/ca16_read/new_conditioned.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '_' -f 3 "$FILE" > aux5
    cut -d ' ' -f 1 "$FILE" > uttid1

    # take everything in aux $0, add that before to the $0 -> aux2
    awk '{print "downloads/African_Accented_French/speech/train/ca16/"$0}' aux5 > aux6

    # aux2/uttid -> aux3
    paste -d "/"  aux6 uttid1 > aux7
    awk '{print $0".wav"}' aux7 > aux8
    paste  -d " " uttid1 aux8 > auxwav2

    # identity function
    paste  -d " " uttid1 uttid1 > auxutt2

    # train yaounde read
    head -6299 ${ACCENTED_FR}/African_Accented_French/transcripts/train/yaounde/fn_text.txt > d_aux
    tail -n +2 d_aux > ${ACCENTED_FR}/African_Accented_French/transcripts/train/yaounde/fn_read_text.txt
    FILE=${ACCENTED_FR}/African_Accented_French/transcripts/train/yaounde/fn_read_text.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '/' -f 9 "$FILE" > aux5
    cut -d ' ' -f 1 aux5 > aux6

    cut -d '/' -f 8-9 "$FILE" > aux9
    cut -d ' ' -f 1 aux9 > aux10

    cut -c -14 aux6 > aux7
    awk '{print "read-"$0}' aux7 > uttid3

    cut -d ' ' -f 2- "$FILE" > aux8
    paste -d ' ' uttid3 aux8 > auxtext3

    awk '{print "downloads/African_Accented_French/speech/train/yaounde/read/"$0}' aux10 > aux11
    paste -d ' ' uttid3 aux11 > auxwav3

    paste -d ' ' uttid3 uttid3 > auxutt3

    # train yaounde answers
    tail -2098 ${ACCENTED_FR}/African_Accented_French/transcripts/train/yaounde/fn_text.txt >  ${ACCENTED_FR}/African_Accented_French/transcripts/train/yaounde/fn_answers_text.txt
    FILE=${ACCENTED_FR}/African_Accented_French/transcripts/train/yaounde/fn_answers_text.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '/' -f 8 "$FILE" > aux5
    cut -d ' ' -f 1 aux5 > aux6

    cut -d '/' -f 7-8 "$FILE" > aux9
    cut -d ' ' -f 1 aux9 > aux10

    cut -c -13 aux6 > aux7
    awk '{print "answers-"$0}' aux7 > uttid4

    cut -d ' ' -f 2- "$FILE" > aux8
    paste -d ' ' uttid4 aux8 > auxtext4

    awk '{print "downloads/African_Accented_French/speech/train/yaounde/answers/"$0}' aux10 > aux11
    paste -d ' ' uttid4 aux11 > auxwav4

    paste -d ' ' uttid4 uttid4 > auxutt4

    # cat everything
    mkdir -p data/train

    cat auxtext1 ${ACCENTED_FR}/African_Accented_French/transcripts/train/ca16_read/new_conditioned.txt auxtext3 auxtext4 > data/train/text
    cat auxwav1 auxwav2 auxwav3 auxwav4 > data/train/wav.scp
    cat auxutt1 auxutt2 auxutt3 auxutt4 > data/train/utt2spk
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

    ./utils/fix_data_dir.sh data/train/

    rm d_aux aux5 aux6 aux7 aux8 aux9 aux10 aux11 auxtext1 auxtext3 auxtext4 auxwav1 auxwav2 auxwav3 auxwav4 auxutt1 auxutt2 auxutt3 auxutt4 uttid1 uttid2 uttid3 uttid4
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "sub-stage 2: Preparing Data for dev"
    mkdir -p data/dev

    FILE=${ACCENTED_FR}/African_Accented_French/transcripts/dev/niger_west_african_fr/transcripts.txt

    cut -d '/' -f 3 "$FILE" > aux5
    cut -d ' ' -f 1 aux5 > aux6

    cut -d '/' -f 2-3 "$FILE" > aux9
    cut -d ' ' -f 1 aux9 > aux10

    cut -c -16 aux6 > uttid

    cut -d ' ' -f 2- "$FILE" > aux8
    paste -d ' ' uttid aux8 > data/dev/text

    awk '{print "downloads/African_Accented_French/speech/dev/niger_west_african_fr/"$0}' aux10 > aux11
    paste -d ' ' uttid aux11 > data/dev/wav.scp

    paste -d ' ' uttid uttid > data/dev/utt2spk

    utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt

    ./utils/fix_data_dir.sh data/dev/

    rm aux5 aux6 aux9 aux10 uttid aux8 aux11

fi

# test; normalization of the test set is done in normalize_test.py
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "sub-stage 3: Preparing Data for test"
    mkdir -p data/test
    FILE=${ACCENTED_FR}/African_Accented_French/transcripts/test/ca16/new_prompts.txt

    cp "$FILE" data/test/text

    cut -d ' ' -f 1 "$FILE" > aux5
    cut -d '_' -f 1-3 aux5 > aux6

    awk '{print "downloads/African_Accented_French/speech/test/ca16/"$0"/"}' aux6 > aux7
    paste -d '' aux7 aux5 > aux8
    awk '{print $0".wav"}' aux8 > aux9
    paste -d ' ' aux5 aux9 > data/test/wav.scp

    paste -d ' ' aux5 aux5 > data/test/utt2spk

    utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

    ./utils/fix_data_dir.sh data/test/

    rm aux5 aux6 aux7 aux8 aux9

fi


# devtest
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "sub-stage 4: Preparing Data for devtest"
    mkdir -p data/devtest

    FILE=${ACCENTED_FR}/African_Accented_French/transcripts/devtest/ca16_read/new_conditioned.txt

    # .split('_') starts at 1, put in aux file, aux is the folder
    cut -d '_' -f 3 "$FILE" > aux
    cut -d ' ' -f 1 "$FILE" > uttid

    # take everything in aux $0, add that before to the $0 -> aux2
    awk '{print "downloads/African_Accented_French/speech/devtest/ca16/"$0}' aux > aux2

    # aux2/uttid -> aux3
    paste -d "/"  aux2 uttid > aux3
    awk '{print $0".wav"}' aux3 > aux4
    paste  -d " " uttid aux4 > data/devtest/wav.scp

    cp ${ACCENTED_FR}/African_Accented_French/transcripts/devtest/ca16_read/new_conditioned.txt data/devtest/text

    # identity function
    paste  -d " " uttid uttid  > data/devtest/utt2spk
    utils/utt2spk_to_spk2utt.pl data/devtest/utt2spk > data/devtest/spk2utt

    ./utils/fix_data_dir.sh data/devtest/
    rm aux aux2 aux3 aux4 uttid
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
