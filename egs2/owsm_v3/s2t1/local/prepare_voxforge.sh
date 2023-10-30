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
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    for lang in it de en es fr nl pt ru; do
        pwd=${PWD}
        cd ../../voxforge/asr1/
        ./local/data.sh --lang ${lang} || exit 1;
        cd ${pwd}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p data/voxforge
    utt_extra_files="text.prev text.ctc"

    declare -A langid_map
    langid_map=(
        [it]='ita' \
        [de]='deu' \
        [en]='eng' \
        [es]='spa' \
        [fr]='fra' \
        [nl]='nld' \
        [pt]='por' \
        [ru]='rus' )
    
    for split in tr dt et; do
        for old_lang in ${!langid_map[@]}; do
            dataset=${split}_${old_lang}
            utils/fix_data_dir.sh ../../voxforge/asr1/data/${dataset}
            python3 local/kaldi_to_whisper.py \
                --data_dir ../../voxforge/asr1/data/${dataset} \
                --output_dir data/voxforge/${dataset}_whisper \
                --prefix VOXFORGE \
                --src ${langid_map[${old_lang}]} \
                --src_field 0 \
                --num_proc 10 \
                --lower_case || exit 1;
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
                data/voxforge/${dataset}_whisper
        done
        utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
            data/voxforge/${split} \
            data/voxforge/${split}_{it,de,en,es,fr,nl,pt,ru}_whisper || exit 1;
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
