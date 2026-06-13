#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

. utils/parse_options.sh || exit 1;

train_set="train_ml_superb2_lang"
dev_set="dev_ml_superb2_lang"
dev_dialect_set="dev_dialect_ml_superb2_lang"
partitions="${train_set} ${dev_set} ${dev_dialect_set}"

python local/prepare_ml_superb2.py
python local/fix_ml_superb2_dev_dialect.py

for x in ${partitions}; do

    utils/utt2spk_to_spk2utt.pl data/${x}/utt2lang > data/${x}/lang2utt
    cp data/${x}/lang2utt data/${x}/category2utt

    mv data/${x}/utt2lang data/${x}/utt2spk
    mv data/${x}/lang2utt data/${x}/spk2utt
    utils/fix_data_dir.sh data/${x} || exit 1;
    utils/validate_data_dir.sh --no-feats --non-print --no-text data/${x} || exit 1;
    mv data/${x}/utt2spk data/${x}/utt2lang
    mv data/${x}/spk2utt data/${x}/lang2utt
done
