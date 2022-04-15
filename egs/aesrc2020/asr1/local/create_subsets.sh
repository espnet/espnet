#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

data=$1     # data transformed into kaldi format

 # divide development set for cross validation
 if [ -d ${data} ];then
     for i in US UK IND CHN JPN PT RU KR CA ES;do
         ./utils/subset_data_dir.sh --spk-list local/files/cvlist/${i}_cv_spk $data/data_all $data/cv/$i
         cat $data/cv/$i/feats.scp >> $data/cv.scp
     done
     ./utils/filter_scp.pl --exclude $data/cv.scp $data/data_all/feats.scp > $data/train_and_dev.scp
     #95-5 split for dev set
     sed -n '0~20p' $data/train_and_dev.scp > $data/dev.scp
     ./utils/filter_scp.pl --exclude $data/dev.scp $data/train_and_dev.scp > $data/train.scp
     ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/train_org
     ./utils/subset_data_dir.sh --utt-list $data/dev.scp $data/data_all $data/dev_org
     ./utils/subset_data_dir.sh --utt-list $data/cv.scp $data/data_all $data/cv_all
 fi

echo "local/subset_data.sh succeeded"
exit 0;
