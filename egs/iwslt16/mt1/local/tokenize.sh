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
echo "tokenize training data"
normalize-punctuation.perl -l en < ${data}/en-${tgt_lang}/train.raw.en | tokenizer.perl -a -l en > ${dumpdir}/${train_set}/train.tkn.en
normalize-punctuation.perl -l ${tgt_lang} < ${data}/en-${tgt_lang}/train.raw.${tgt_lang}  | tokenizer.perl -a -l ${tgt_lang} > ${dumpdir}/${train_set}/train.tkn.${tgt_lang}

# validation
echo "tokenize validation data"
normalize-punctuation.perl -l en < ${data}/en-${tgt_lang}/tst2012.raw.en | tokenizer.perl -a -l en > ${dumpdir}/${train_dev}/tst2012.tkn.en
normalize-punctuation.perl -l ${tgt_lang}  < ${data}/en-${tgt_lang}/tst2012.raw.${tgt_lang}  | tokenizer.perl -a -l ${tgt_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.${tgt_lang}


# test
echo "tokenize test data"
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    normalize-punctuation.perl -l en < ${data}/en-${tgt_lang}/${name}.raw.en | tokenizer.perl -a -l en > ${dumpdir}/${ts}/${name}.tkn.en
    normalize-punctuation.perl -l ${tgt_lang} < ${data}/en-${tgt_lang}/${name}.raw.${tgt_lang} | tokenizer.perl -a -l ${tgt_lang} > ${dumpdir}/${ts}/${name}.tkn.${tgt_lang}
done

