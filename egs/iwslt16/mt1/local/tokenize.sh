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

# train
echo "tokenize training data"
normalize-punctuation.perl -l ${src_lang} < ${data}/${src_lang}-${tgt_lang}/train.raw.${src_lang} | tokenizer.perl -a -l ${src_lang} > ${dumpdir}/${train_set}/train.tkn.${src_lang}
normalize-punctuation.perl -l ${tgt_lang} < ${data}/${src_lang}-${tgt_lang}/train.raw.${tgt_lang}  | tokenizer.perl -a -l ${tgt_lang} > ${dumpdir}/${train_set}/train.tkn.${tgt_lang}

# validation
echo "tokenize validation data"
normalize-punctuation.perl -l ${src_lang} < ${data}/${src_lang}-${tgt_lang}/tst2012.raw.${src_lang} | tokenizer.perl -a -l ${src_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.${src_lang}
normalize-punctuation.perl -l ${tgt_lang}  < ${data}/${src_lang}-${tgt_lang}/tst2012.raw.${tgt_lang}  | tokenizer.perl -a -l ${tgt_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.${tgt_lang}


# test
echo "tokenize test data"
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    normalize-punctuation.perl -l ${src_lang} < ${data}/${src_lang}-${tgt_lang}/${name}.raw.${src_lang} | tokenizer.perl -a -l ${src_lang} > ${dumpdir}/${ts}/${name}.tkn.${src_lang}
    normalize-punctuation.perl -l ${tgt_lang} < ${data}/${src_lang}-${tgt_lang}/${name}.raw.${tgt_lang} | tokenizer.perl -a -l ${tgt_lang} > ${dumpdir}/${ts}/${name}.tkn.${tgt_lang}
done
