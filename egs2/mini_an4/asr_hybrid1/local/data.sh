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
stop_stage=100

an4_root=./downloads/an4

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Untar downloads.tar.gz"
    if [ ! -e downloads/ ]; then
        tar -xvf downloads.tar.gz
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{train,test}

    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi

    python3 local/data_prep.py "${an4_root}" sph2pipe

    for x in test train; do
        for f in text wav.scp utt2spk; do
            sort "data/${x}/${f}" -o "data/${x}/${f}"
        done
        utils/utt2spk_to_spk2utt.pl "data/${x}/utt2spk" > "data/${x}/spk2utt"
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: prepare data/local"
    rm -rf data/local/{dict,lang}
    mkdir -p data/local/{dict,lang}

    python local/lexicon_prep.py "${an4_root}"
    echo '<UNK> SIL' >> data/local/dict/lexicon.txt
    <"${an4_root}"/etc/an4.phone grep -v 'SIL' > data/local/dict/nonsilence_phones.txt
    echo 'SIL' > data/local/dict/silence_phones.txt
    echo 'SIL' > data/local/dict/optional_silence.txt
    echo -n > data/local/dict/extra_questions.txt

    utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
    arpa2fst "${an4_root}"/etc/an4.ug.lm | fstprint | \
        grep -v 'HALL\|LANE\|MEMORY\|TWELVTH\|WEAN' | \
        utils/remove_oovs.pl data/lang/oov.txt | utils/eps2disambig.pl | utils/s2eps.pl | \
        fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt \
            --keep_isymbols=false --keep_osymbols=false | \
        fstarcsort > data/lang/G.fst
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
