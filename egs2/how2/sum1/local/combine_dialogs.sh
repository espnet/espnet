#! /bin/bash


dset=

. utils/parse_options.sh || exit 1



mkdir -p data/${dset}_vid

cp data/${dset}/wav.scp data/${dset}_vid/
cat data/${dset}/wav.scp | awk -F ' ' '{print $1,$1}' | LC_ALL=C sort > data/${dset}_vid/utt2spk
cat data/${dset}/text | python local/combine_utts.py | LC_ALL=C sort > data/${dset}_vid/text
utils/utt2spk_to_spk2utt.pl data/${dset}_vid/utt2spk > data/${dset}_vid/spk2utt
utils/fix_data_dir.sh data/${dset}_vid
