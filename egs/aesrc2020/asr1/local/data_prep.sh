#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

raw_data=$1     # raw data with metadata, txt and wav
data=$2         # data transformed into kaldi format

# generate kaldi format data for all
if [ -d ${raw_data} ];then 
    echo "Generating kaldi format data."
    mkdir -p $data/data_all
    find $raw_data -type f -name "*.wav" > $data/data_all/wavpath
    awk -F'/' '{print $(NF-2)"-"$(NF-1)"-"$NF}' $data/data_all/wavpath | sed 's:\.wav::g' > $data/data_all/uttlist
    paste $data/data_all/uttlist $data/data_all/wavpath > $data/data_all/wav.scp
    python local/preprocess.py $data/data_all/wav.scp $data/data_all/trans $data/data_all/utt2spk # faster than for in shell
    ./utils/utt2spk_to_spk2utt.pl $data/data_all/utt2spk > $data/data_all/spk2utt
fi

# clean transcription
if [ -d $data/data_all ];then
    echo "Cleaning transcription."
    tr '[a-z]' '[A-Z]' < $data/data_all/trans > $data/data_all/trans_upper
    # turn "." in specific abbreviations into "<m>" tag
    sed -i -e 's: MR\.: MR<m>:g' -e 's: MRS\.: MRS<m>:g' -e 's: MS\.: MS<m>:g' \
        -e 's:^MR\.:MR<m>:g' -e 's:^MRS\.:MRS<m>:g' -e 's:^MS\.:MS<m>:g' $data/data_all/trans_upper 
	# fix bug
    sed -i 's:^ST\.:STREET:g' $data/data_all/trans_upper 
    sed -i 's: ST\.: STREET:g' $data/data_all/trans_upper 
    # punctuation marks
    sed -i "s%,\|\.\|?\|!\|;\|-\|:\|,'\|\.'\|?'\|!'\| '% %g" $data/data_all/trans_upper
    sed -i 's:<m>:.:g' $data/data_all/trans_upper
    # blank
    sed -i 's:[ ][ ]*: :g' $data/data_all/trans_upper
    paste $data/data_all/uttlist $data/data_all/trans_upper > $data/data_all/text

    # critally, must replace tab with space between uttid and text
    sed -e "s/\t/ /g" -i $data/data_all/text
fi

echo "local/data_prep.sh succeeded"
exit 0;
