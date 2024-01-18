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
stop_stage=3
nproc=64

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

utt_extra_files="text.prev text.ctc"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Googleil18n datasets contains multiple datasets.
    # This script only uses those from OpenSLR
    pwd=${PWD}
    cd ../../open_li110/asr1/
    ./local/data.sh \
      --voxforge_lang "" \
      --commonvoice_lang "" \
      --mls_lang "" \
      --voxpopuli_lang "" \
      --extra_langs "" \
      --lid false
    cd ${pwd}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # convert to owsm style dataset
    mkdir -p data/openslr

    # ISO-639-3 (from local/cv-iso-693-3.txt)
    declare -A langid_map
    langid_map=(
        [af_openslr32]='afr' \
        [bn_bd_openslr37]='ben' \
        [bn_in_openslr37]='ben' \
        [bn_openslr53]='ben' \
        [ca_openslr69]='cat' \
        [es_openslr61]='spa' \
        [es_openslr71]='spa' \
        [es_openslr72]='spa' \
        [es_openslr73]='spa' \
        [es_openslr74]='spa' \
        [es_openslr75]='spa' \
        [eu_openslr76]='eus' \
        [gl_openslr77]='glg' \
        [gu_openslr78]='guj' \
        [jv_openslr35]='jav' \
        [jv_openslr41_female]='jav' \
        [jv_openslr41_male]='jav' \
        [km_openslr42_male]='khm' \
        [kn_openslr79]='kan' \
        [ml_openslr63]='mal' \
        [mr_openslr64]='mar' \
        [ne_openslr43_female]='nep' \
        [ne_openslr54]='nep' \
        [si_openslr52]='sin' \
        [st_openslr32]='sot' \
        [su_openslr44_female]='sun' \
        [su_openslr44_male]='sun' \
        [ta_openslr65]='tam' \
        [te_openslr66]='tel' \
        [tn_openslr32]='tsn' \
        [xh_openslr32]='xho' \
        [yo_openslr86]='yor' \
        [su_openslr36]='sun' \
    )

    for lang in ${!langid_map[@]}; do
        for part in train dev test; do
            echo "processing ${part}_${lang}"
            utils/fix_data_dir.sh ../../open_li110/asr1/data/${part}_${lang}
            python3 local/kaldi_to_whisper.py \
                --data_dir ../../open_li110/asr1/data/${part}_${lang} \
                --output_dir data/openslr/${part}_${lang} \
                --prefix OpenSLR \
                --src ${langid_map[${lang}]} \
                --src_field 0 \
                --num_proc ${nproc} || exit 1;
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}"  \
                data/openslr/${part}_${lang}
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # combine train and valid
    utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
        data/openslr/train data/openslr/train_*openslr* || exit 1;
    utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
        data/openslr/dev data/openslr/dev_*openslr* || exit 1;
    utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
        data/openslr/test data/openslr/test_*openslr* || exit 1;
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
