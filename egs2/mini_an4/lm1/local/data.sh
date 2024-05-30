#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# Adapted from asr1 task
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

train_set="train_nodev"
train_dev="train_dev"

data_dir="data/speech"
task_dirs="asr tts"

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

    python3 local/data_prep.py ${an4_root} sph2pipe

    # process data per task
    for task_dir in ${task_dirs}; do
        for x in test train; do
            mkdir -p ${data_dir}/${task_dir}/${x}
            for f in text wav.scp utt2spk; do
                sort data/${x}/${f} -o ${data_dir}/${task_dir}/${x}/${f}
            done
            utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > ${data_dir}/${task_dir}/${x}/spk2utt
        done

        # make a dev set
        utils/subset_data_dir.sh --first ${data_dir}/${task_dir}/train 1 ${data_dir}/${task_dir}/${train_dev}
        n=$(($(wc -l < ${data_dir}/${task_dir}/train/text) - 1))
        utils/subset_data_dir.sh --last ${data_dir}/${task_dir}/train ${n} ${data_dir}/${task_dir}/${train_set}

        # Create "test_seg" in order to test the use case of segments
        rm -rf ${data_dir}/${task_dir}/test_seg
        utils/copy_data_dir.sh ${data_dir}/${task_dir}/test ${data_dir}/${task_dir}/test_seg
        <${data_dir}/${task_dir}/test/wav.scp awk '{ for(i=2;i<=NF;i++){a=a " " $i}; print($1 "_org", a) }' > ${data_dir}/${task_dir}/test_seg/wav.scp
        cat << EOF > ${data_dir}/${task_dir}/test_seg/segments
fcaw-cen8-b fcaw-cen8-b_org 0.0 2.9
mmxg-cen8-b mmxg-cen8-b_org 0.0 2.3
EOF
        rm -rf ${data_dir}/${task_dir}/train

    done
    rm -rf data/{train,test}

fi

for x in "${train_set}" "${train_dev}" "test" "test_seg"; do
    mkdir -p data/${x}/speech
    for task_dir in ${task_dirs}; do
        mv ${data_dir}/${task_dir}/${x} data/${x}/speech/${task_dir}
    done
done

log "Successfully finished. [elapsed=${SECONDS}s]"
