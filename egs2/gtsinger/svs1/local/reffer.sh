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
fs=None
g2p=pyopenjtalk

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

train_set="tr_no_dev"
train_dev="dev"
recog_set="eval"

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
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare segments"
    for x in ${train_set} ${train_dev} ${recog_set}; do
        src_data=data/${x}

        local/prep_segments.py --silence pau --silence sil --silence br --silence GlottalStop --silence Edge ${src_data}
        mv ${src_data}/segments.tmp ${src_data}/segments
        mv ${src_data}/label.tmp ${src_data}/label
        mv ${src_data}/utt2spk.temp ${src_data}/utt2spk

        local/prep_segments_from_xml.py --silence P --silence B ${src_data}
        mv ${src_data}/text.tmp ${src_data}/text
        mv ${src_data}/segments_from_xml.tmp ${src_data}/segments_from_xml
        mv ${src_data}/score.scp.tmp ${src_data}/score.scp
        utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" ${src_data}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Check alignments"
    # We align music info at phone level if both annotation (label) and music score are used.
    for x in ${train_set} ${train_dev} ${recog_set}; do
        src_data=data/${x}
        pyscripts/utils/check_align.py ${src_data} --g2p ${g2p}
        mv ${src_data}/score.scp.tmp ${src_data}/score.scp
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" ${src_data}
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"

# ./run.sh --stage 1 --stop_stage 1

