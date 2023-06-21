#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

data=$1     # data transformed into kaldi format
recog_set=$2

 # divide development set for cross validation
 if [ -d ${data} ];then
     for i in ${recog_set};do
         ./utils/subset_data_dir.sh --spk-list local/files/cvlist/${i}_cv_spk $data/data_all $data/cv/$i
         cat $data/cv/$i/wav.scp >> $data/cv.scp
     done
     ./utils/filter_scp.pl --exclude $data/cv.scp $data/data_all/wav.scp > $data/train_and_dev.scp
     #95-5 split for dev set
     sed -n '0~20p' $data/train_and_dev.scp > $data/dev.scp
     ./utils/filter_scp.pl --exclude $data/dev.scp $data/train_and_dev.scp > $data/train.scp
     ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/train
     ./utils/subset_data_dir.sh --utt-list $data/dev.scp $data/data_all $data/valid
     ./utils/subset_data_dir.sh --utt-list $data/cv.scp $data/data_all $data/test
 fi

echo "local/subset_data.sh succeeded"