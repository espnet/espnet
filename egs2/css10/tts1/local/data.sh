#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=100
text_format=raw
langs="de el es fi fr hu ja nl ru zh"
threshold=35
nj=32

log "$0 $*"
# shellcheck disable=SC1091
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

# shellcheck disable=SC1091
. ./path.sh || exit 1;
# shellcheck disable=SC1091
. ./cmd.sh || exit 1;
# shellcheck disable=SC1091
. ./db.sh || exit 1;

if [ -z "${CSS10}" ]; then
   log "Fill the value of 'CSS10' of db.sh"
   exit 1
fi

db_root=${CSS10}
train_set=tr_no_dev
dev_set=dev
eval_set=eval1

if [ ! -e "${CSS10}" ]; then
    log "CSS10 dataset is not found."
    log "Please download it from https://github.com/Kyubyong/css10 and locate as follows:"
    cat << EOF
$ vim db.sh
CSS10=/path/to/CSS10

$ tree -L 2 /path/to/CSS10
├── de
│   ├── achtgesichterambiwasse
│   ├── meisterfloh
│   ├── serapionsbruederauswahl
│   └── transcript.txt
├── el
│   ├── Paramythi_horis_onoma
│   └── transcript.txt
├── es
│   ├── 19demarzo
│   ├── bailen
│   ├── batalla_arapiles
│   └── transcript.txt
├── fi
│   ├── ensimmaisetnovellit
│   ├── gulliverin_matkat_kaukaisilla_mailla
│   ├── kaleri-orja
│   ├── salmelan_heinatalkoot
│   └── transcript.txt
├── fr
│   ├── lesmis
│   ├── lupincontresholme
│   └── transcript.txt
├── hu
│   ├── egri_csillagok
│   └── transcript.txt
├── ja
│   ├── meian
│   └── transcript.txt
├── nl
│   ├── 20000_mijlen
│   └── transcript.txt
├── ru
│   ├── early_short_stories
│   ├── icemarch
│   ├── shortstories_childrenadults
│   └── transcript.txt
└── zh
    ├── call_to_arms
    ├── chao_hua_si_she
    └── transcript.txt
EOF
    exit 1
fi

# define g2p dict
declare -A g2p_dict=(
    ["de"]="espeak_ng_german"
    ["el"]="espeak_ng_greek"
    ["es"]="espeak_ng_spanish"
    ["fi"]="espeak_ng_finnish"
    ["fr"]="espeak_ng_french"
    ["hu"]="espeak_ng_hungarian"
    ["ja"]="pyopenjtalk"
    ["nl"]="espeak_ng_dutch"
    ["ru"]="espeak_ng_russian"
    ["zh"]="pypinyin_g2p_phone"
)

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    for lang in ${langs}; do
        local/data_prep.sh "${db_root}/${lang}" "data/${lang}"
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: scripts/audio/trim_silence.sh"
    for lang in ${langs}; do
        # shellcheck disable=SC2154
        scripts/audio/trim_silence.sh \
            --cmd "${train_cmd}" \
            --nj "${nj}" \
            --fs 22050 \
            --win_length 1024 \
            --shift_length 256 \
            --threshold "${threshold}" \
            "data/${lang}" "data/${lang}/log"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${text_format}" = phn ]; then
    log "stage 2: pyscripts/utils/convert_text_to_phn.py"
    for lang in ${langs}; do
        g2p=${g2p_dict[${lang}]}
        utils/copy_data_dir.sh "data/${lang}" "data/${lang}_phn"
        pyscripts/utils/convert_text_to_phn.py \
            --g2p "${g2p}" --nj "${nj}" \
            "data/${lang}/text" "data/${lang}_phn/text"
        utils/fix_data_dir.sh "data/${lang}_phn"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: utils/subset_data_dir.sh"
    suffix=""
    if [ "${text_format}" = phn ]; then
        suffix="_phn"
    fi
    combine_train_dirs=()
    combine_dev_dirs=()
    combine_eval_dirs=()
    for lang in ${langs}; do
        utils/subset_data_dir.sh "data/${lang}${suffix}" 100 "data/${lang}_deveval${suffix}"
        utils/subset_data_dir.sh --first "data/${lang}_deveval${suffix}" 50 "data/${lang}_${dev_set}${suffix}"
        utils/subset_data_dir.sh --last "data/${lang}_deveval${suffix}" 50 "data/${lang}_${eval_set}${suffix}"
        utils/copy_data_dir.sh "data/${lang}${suffix}" "data/${lang}_${train_set}${suffix}"
        utils/filter_scp.pl --exclude "data/${lang}_deveval${suffix}/wav.scp" \
            "data/${lang}${suffix}/wav.scp" > "data/${lang}_${train_set}${suffix}/wav.scp"
        utils/fix_data_dir.sh "data/${lang}_${train_set}${suffix}"
        combine_train_dirs+=("data/${lang}_${train_set}${suffix}")
        combine_dev_dirs+=("data/${lang}_${dev_set}${suffix}")
        combine_eval_dirs+=("data/${lang}_${eval_set}${suffix}")
    done
    utils/combine_data.sh "data/${train_set}${suffix}" "${combine_train_dirs[@]}"
    utils/combine_data.sh "data/${dev_set}${suffix}" "${combine_dev_dirs[@]}"
    utils/combine_data.sh "data/${eval_set}${suffix}" "${combine_eval_dirs[@]}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
