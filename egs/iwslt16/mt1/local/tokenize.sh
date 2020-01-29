#!/bin/bash

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


. ./path.sh || exit 1

data=$1
tgt_lang=$2
dumpdir=$3
train_set=$4
train_dev=$5
trans_set=$6

# train
tokenizer.perl -a -l en < ${data}/en-${tgt_lang}/train.raw.en > ${dumpdir}/${train_set}/train.tkn.en
tokenizer.perl -a -l ${tgt_lang} < ${data}/en-${tgt_lang}/train.raw.${tgt_lang} > ${dumpdir}/${train_set}/train.tkn.${tgt_lang}

# validation
tokenizer.perl -a -l en < ${data}/en-${tgt_lang}/tst2012.raw.en > ${dumpdir}/${train_dev}/tst2012.tkn.en
tokenizer.perl -a -l ${tgt_lang} < ${data}/en-${tgt_lang}/tst2012.raw.${tgt_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.${tgt_lang}


# test
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    tokenizer.perl -a -l en < ${data}/en-${tgt_lang}/${name}.raw.en > ${dumpdir}/${ts}/${name}.tkn.en
    tokenizer.perl -a -l ${tgt_lang} < ${data}/en-${tgt_lang}/${name}.raw.${tgt_lang} > ${dumpdir}/${ts}/${name}.tkn.${tgt_lang}
done



