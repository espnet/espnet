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

nproc=64
stage=1
stop_stage=3

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

train_sets="101 102 103 104 105 106 107 201 202 203 204 205 206 207 301 302 303 304 305 306 307 401 402 403 404"
valid_sets="107 201 307 404"
utt_extra_files="text.prev text.ctc"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # use the original espnet scripts
    pwd=${PWD}
    cd ../../babel/asr1/
    ./local/data.sh
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # convert to owsm style dataset
    mkdir -p data/babel
    cp ../../babel/asr1/data/nlsym.txt ./data/babel/nlsyms.txt
    nlsyms=$(cat data/babel/nlsyms.txt | tr '\n' ' ')

    declare -A langid_map
    langid_map=(
        [101]='yue' \
        [102]='asm' \
        [103]='ben' \
        [104]='pus' \
        [105]='tur' \
        [106]='tgl' \
        [107]='vie' \
        [201]='hat' \
        [202]='swa' \
        [203]='lao' \
        [204]='tam' \
        [205]='kmr' \
        [206]='zul' \
        [207]='tpi' \
        [301]='ceb' \
        [302]='kaz' \
        [303]='tel' \
        [304]='lit' \
        [305]='grn' \
        [306]='ibo' \
        [307]='amh' \
        [401]='mon' \
        [402]='jav' \
        [403]='luo' \
        [404]='kat' )

    # Train part
    for part in ${train_sets}; do
        utils/fix_data_dir.sh ../../babel/asr1/data/${part}/data/train_${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../babel/asr1/data/${part}/data/train_${part} \
            --output_dir data/babel/train_${part}_whisper \
            --prefix BABEL \
            --src ${langid_map[${part}]} \
            --src_field 6 \
            --num_proc ${nproc} \
            --nlsyms ${nlsyms} \
            --lower_case || exit 1;
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
            data/babel/train_${part}_whisper
    done

    # Dev part
    for part in ${valid_sets}; do
        utils/fix_data_dir.sh ../../babel/asr1/data/${part}/data/dev_${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../babel/asr1/data/${part}/data/dev_${part} \
            --output_dir data/babel/dev_${part}_whisper \
            --prefix BABEL \
            --src ${langid_map[${part}]} \
            --src_field 6 \
            --num_proc ${nproc} \
            --nlsyms ${nlsyms} \
            --lower_case || exit 1;
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
            data/babel/dev_${part}_whisper
    done

    # Test part
    for part in ${valid_sets}; do
        utils/fix_data_dir.sh ../../babel/asr1/data/${part}/data/eval_${part}
        python3 local/kaldi_to_whisper.py \
            --data_dir ../../babel/asr1/data/${part}/data/eval_${part} \
            --output_dir data/babel/test_${part}_whisper \
            --prefix BABEL \
            --src ${langid_map[${part}]} \
            --src_field 6 \
            --num_proc ${nproc} \
            --nlsyms ${nlsyms} \
            --lower_case || exit 1;
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
            data/babel/test_${part}_whisper
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # combine
    utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
        data/babel/train data/babel/train_*_whisper || exit 1;
    utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
        data/babel/dev data/babel/dev_*_whisper || exit 1;
    utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
        data/babel/test data/babel/test_*_whisper || exit 1;
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
