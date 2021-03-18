#!/usr/bin/env bash

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1

data=$1
src_lang=$2
tgt_lang=$3
dumpdir=$4
train_set=$5
train_dev=$6
trans_set=$7


# train truecaser model
train-truecaser.perl --model /tmp/tc_model.${src_lang} --corpus ${dumpdir}/${train_set}/train.tkn.${src_lang}
train-truecaser.perl --model /tmp/tc_model.${tgt_lang} --corpus ${dumpdir}/${train_set}/train.tkn.${tgt_lang}


# apply truecasing to train
truecase.perl --model /tmp/tc_model.${src_lang} < ${dumpdir}/${train_set}/train.tkn.${src_lang} > ${dumpdir}/${train_set}/train.tkn.tc.${src_lang}
truecase.perl --model /tmp/tc_model.${tgt_lang} < ${dumpdir}/${train_set}/train.tkn.${tgt_lang} > ${dumpdir}/${train_set}/train.tkn.tc.${tgt_lang}

# apply truecasing to dev
truecase.perl --model /tmp/tc_model.${src_lang} < ${dumpdir}/${train_dev}/tst2012.tkn.${src_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.tc.${src_lang}
truecase.perl --model /tmp/tc_model.${tgt_lang} < ${dumpdir}/${train_dev}/tst2012.tkn.${tgt_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.tc.${tgt_lang}

# apply truecasing to test
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    truecase.perl --model /tmp/tc_model.${src_lang} < ${dumpdir}/${ts}/${name}.tkn.${src_lang} > ${dumpdir}/${ts}/${name}.tkn.tc.${src_lang}
    truecase.perl --model /tmp/tc_model.${tgt_lang} < ${dumpdir}/${ts}/${name}.tkn.${tgt_lang} > ${dumpdir}/${ts}/${name}.tkn.tc.${tgt_lang}
done
