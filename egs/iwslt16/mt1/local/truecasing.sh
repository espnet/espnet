#!/bin/bash

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -x
set -e

. ./path.sh || exit 1

data=$1
tgt_lang=$2
dumpdir=$3
train_set=$4
train_dev=$5
trans_set=$6


# train truecaser model
train-truecaser.perl --model /tmp/tc_model.en --corpus ${dumpdir}/${train_set}/train.tkn.en
train-truecaser.perl --model /tmp/tc_model.${tgt_lang} --corpus ${dumpdir}/${train_set}/train.tkn.${tgt_lang}


# apply truecasing to train
truecase.perl --model /tmp/tc_model.en < ${dumpdir}/${train_set}/train.tkn.en > ${dumpdir}/${train_set}/train.tkn.tc.en
truecase.perl --model /tmp/tc_model.${tgt_lang} < ${dumpdir}/${train_set}/train.tkn.${tgt_lang} > ${dumpdir}/${train_set}/train.tkn.tc.${tgt_lang}

# apply truecasing to dev
truecase.perl --model /tmp/tc_model.en < ${dumpdir}/${train_dev}/tst2012.tkn.en > ${dumpdir}/${train_dev}/tst2012.tkn.tc.en
truecase.perl --model /tmp/tc_model.${tgt_lang} < ${dumpdir}/${train_dev}/tst2012.tkn.${tgt_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.tc.${tgt_lang}

# apply truecasing to test
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    truecase.perl --model /tmp/tc_model.en < ${dumpdir}/${ts}/${name}.tkn.en > ${dumpdir}/${ts}/${name}.tkn.tc.en
    truecase.perl --model /tmp/tc_model.${tgt_lang} < ${dumpdir}/${ts}/${name}.tkn.${tgt_lang} > ${dumpdir}/${ts}/${name}.tkn.tc.${tgt_lang}
done
