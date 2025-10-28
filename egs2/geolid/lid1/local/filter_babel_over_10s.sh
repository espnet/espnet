#!/bin/bash

dump_dir="dump/raw"
splits="train_babel_lang dev_babel_lang"
splits_over_10s="train_babel_over_10s_lang dev_babel_over_10s_lang"

. utils/parse_options.sh || exit 1;

for split in $splits; do
    python local/filter_babel_over_10s.py --babel_dir $dump_dir/$split --babel_over_10s_dir $dump_dir/$split_over_10s
done

for split in $splits_over_10s; do
    cp $dump_dir/$split/audio_format $dump_dir/$split/audio_format
    cp $dump_dir/$split/feats_type $dump_dir/$split/feats_type
    ./utils/utt2spk_to_spk2utt.pl $dump_dir/$split/utt2spk > $dump_dir/$split/spk2utt
    cp $dump_dir/$split/spk2utt $dump_dir/$split/category2utt
done
