#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=100
fs=24000
g2p=pypinyin_g2p_phone_without_prosody

#--    raise NotImplementedError(f"Not supported: g2p_type={g2p_type}")
#--    NotImplementedError: Not supported: g2p_type=None
#g2p_type='phonetisaurus'  # 或者其他支持的类型


log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${GTSINGER}" ]; then
    log "Fill the value of 'GTSINGER' of db.sh"
    exit 1
fi

mkdir -p ${GTSINGER}

train_set="tr_no_dev" #可能表示这是不包含开发集的训练数据，即仅用于训练模型的数据
train_dev="dev" #表示开发集(验证集)，用于调整模型的超参数和监控模型的性能
recog_set="eval" #评估集(测试集)，通常是用于最终模型测试的数据，用来衡量模型在未见过的数据上的表现
test_sets="dev eval"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The ameboshi data should be downloaded from https://parapluie2c56m.wixsite.com/mysite
    # with authentication
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dataset split "
    # We use a pre-defined split (see details in local/dataset_split.py)"
    python local/dataset_split.py ${GTSINGER} \
        data/${train_set} data/${train_dev} data/${recog_set} --fs ${fs}
    
    for x in ${train_set} ${train_dev} ${recog_set}; do
        src_data=data/${x}
        mv ${src_data}/score.scp.tmp ${src_data}/score.scp
        utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" ${src_data}
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"

# ./run.sh --stage 1 --stop_stage 1

