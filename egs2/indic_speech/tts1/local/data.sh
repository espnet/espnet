#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

g2p=espeak_ng_hindi
nj=12
text_format=phn

log "$0 $*"
. utils/parse_options.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

db_root=${INDIC_SPEECH}
spk="Hindi_TTS_dataset"

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    unzip "${db_root}/${spk}/Dataset.zip"
    mv Dataset "${db_root}/${spk}"
    rm "${db_root}/${spk}/Dataset.zip"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Preparing Data"
    python3 local/data_prep.py -d ${db_root}
    utils/spk2utt_to_utt2spk.pl data/${spk}/spk2utt > data/${spk}/utt2spk
    utils/fix_data_dir.sh data/${spk}
    utils/validate_data_dir.sh --no-feats data/${spk}
    utils/subset_data_dir.sh --last data/${spk} 200 data/${spk}_tmp
    utils/subset_data_dir.sh --last data/${spk}_tmp 100 data/${eval_set}
    utils/subset_data_dir.sh --first data/${spk}_tmp 100 data/${dev_set}
    n=$(( $(wc -l < data/${spk}/wav.scp) - 200 ))
    utils/subset_data_dir.sh --first data/${spk} ${n} data/${train_set}
    rm -rf data/${spk}_tmp
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ "${text_format}" = phn ]; then
    log "stage 1: pyscripts/utils/convert_text_to_phn.py"
    for dset in "${train_set}" "${dev_set}" "${eval_set}"; do
        utils/copy_data_dir.sh "data/${dset}" "data/${dset}_phn"
        pyscripts/utils/convert_text_to_phn.py --g2p "${g2p}" --nj "${nj}" \
            "data/${dset}/text" "data/${dset}_phn/text"
        utils/fix_data_dir.sh "data/${dset}_phn"
    done
fi
