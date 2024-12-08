#!/bin/bash


for dset in train_temp dev_temp test_temp urgent_blind_pair ; do

./utils/spk2utt_to_utt2spk.pl data/${dset}/utt2spk > data/${dset}/spk2utt
for file in text utt2spk spk2utt metric.scp wav.scp ref_wav.scp; do
sort -o data/${dset}/${file} data/${dset}/${file}
done

done
