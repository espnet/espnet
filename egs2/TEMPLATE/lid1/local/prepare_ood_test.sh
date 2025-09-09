#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
        echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# parse args
dump_dir=dump/raw
train_set=
test_sets=

. utils/parse_options.sh || exit 1;

if [ -z "${dump_dir}" ] || [ -z "${train_set}" ] || [ -z "${test_sets}" ]; then
    log "Usage: $0 --dump_dir <dump_dir> --train_set <train_set> --test_sets <test_sets>"
    exit 1
fi

# check dir exist
for dir in "${train_set}" ${test_sets}; do
    if [ ! -d "${dump_dir}/${dir}" ]; then
        log "Error: ${dump_dir}/${dir} does not exist."
        exit 1
    fi
done

# cross set names
cross_sets=""
for test_set in ${test_sets}; do
    cross_set="${test_set}_cross_${train_set}"
    cross_sets="${cross_sets} ${cross_set}"
done

read -r -a cross_sets_array <<< "${cross_sets}"
read -r -a test_sets_array <<< "${test_sets}"
test_sets_renew=""
cross_sets_renew=""
for i in "${!cross_sets_array[@]}"; do
    cross_set="${cross_sets_array[$i]}"
    test_set="${test_sets_array[$i]}"

    if [ -d "${dump_dir}/${cross_set}" ]; then
        log "Warning: cross_set ${cross_set} already exists in ${dump_dir}."
        log "Skipping preparation."
        continue
    fi

    test_sets_renew+="${test_set} "
    cross_sets_renew+="${cross_set} "
done

if [ -z "${test_sets_renew}" ]; then
    log "All cross_sets already exist."
    exit 0
fi

python local/prepare_ood_test.py \
    --dump_dir ${dump_dir} \
    --train_set ${train_set} \
    --test_sets "${test_sets_renew}"

for cross_set in ${cross_sets_renew}; do
    ./utils/utt2spk_to_spk2utt.pl ${dump_dir}/${cross_set}/utt2lang > ${dump_dir}/${cross_set}/lang2utt
    cp ${dump_dir}/${cross_set}/lang2utt ${dump_dir}/${cross_set}/category2utt
done

log "Successfully prepared the following OOD test sets: ${cross_sets_renew}"
