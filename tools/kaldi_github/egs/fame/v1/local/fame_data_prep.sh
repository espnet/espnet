#!/bin/bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2017  Radboud University (Author: Emre Yilmaz)

# Apache 2.0

corpus=$1
set -e -o pipefail
if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the FAME! speech database"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "Preparing train, development and test data"
mkdir -p data data/local data/train_asr data/devel_asr data/test_asr

for x in train devel test; do
    echo "Copy spk2utt, utt2spk, wav.scp, text for $x"
    cp $corpus/data/$x/text     data/${x}_asr/text    || exit 1;
    cp $corpus/data/$x/spk2utt  data/${x}_asr/spk2utt || exit 1;
    cp $corpus/data/$x/utt2spk  data/${x}_asr/utt2spk || exit 1;

    # the corpus wav.scp contains physical paths, so we just re-generate
    # the file again from scratchn instead of figuring out how to edit it
    for rec in $(awk '{print $1}' $corpus/data/$x/text) ; do
        spk=${rec%_*}
        filename=$corpus/fame/wav/${x}/${rec:8}.wav
        if [ ! -f "$filename" ] ; then
            echo >&2 "The file $filename could not be found ($rec)"
            exit 1
        fi
        # we might want to store physical paths as a general rule
        filename=$(readlink -f $filename)
        echo "$rec $filename"
    done > data/${x}_asr/wav.scp

    # fix_data_dir.sh fixes common mistakes (unsorted entries in wav.scp,
    # duplicate entries and so on). Also, it regenerates the spk2utt from
    # utt2sp
    utils/fix_data_dir.sh data/${x}_asr
done

echo "Copying language model"
if [ -f $corpus/lm/LM_FR_IKN3G ] ; then
    gzip -c $corpus/lm/LM_FR_IKN3G > data/local/LM.gz
fi

echo "Data preparation completed."

