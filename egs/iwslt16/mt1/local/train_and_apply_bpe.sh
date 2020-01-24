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

if [ -e subword-nmt ]; then
  echo "subword-nmt toolkit already exists. continue..."
else
  git clone https://github.com/rsennrich/subword-nmt
fi

mkdir -p ${dumpdir}/vocab

SRC_BPE_CODE=${dumpdir}/vocab/vocab.en_bpe16000
TRG_BPE_CODE=${dumpdir}/vocab/vocab.${tgt_lang}_bpe16000

# learn bpe merge operation
python subword-nmt/subword_nmt/learn_bpe.py -s 16000 < ${dumpdir}/${train_set}/train.tkn.tc.clean.en > ${SRC_BPE_CODE}
python subword-nmt/subword_nmt/learn_bpe.py -s 16000 < ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang} > ${TRG_BPE_CODE}

# apply bpe operation
# train
python subword-nmt/subword_nmt/apply_bpe.py -c ${SRC_BPE_CODE} < ${dumpdir}/${train_set}/train.tkn.tc.clean.en > ${dumpdir}/${train_set}/train.tkn.tc.clean.en_bpe16000
python subword-nmt/subword_nmt/apply_bpe.py -c ${TRG_BPE_CODE} < ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang} > ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang}_bpe16000


# valid
python subword-nmt/subword_nmt/apply_bpe.py -c ${SRC_BPE_CODE} < ${dumpdir}/${train_dev}/tst2012.tkn.tc.en > ${dumpdir}/${train_dev}/tst2012.tkn.tc.en_bpe16000
python subword-nmt/subword_nmt/apply_bpe.py -c ${TRG_BPE_CODE} < ${dumpdir}/${train_dev}/tst2012.tkn.tc.${tgt_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.tc.${tgt_lang}_bpe16000

# test
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    python subword-nmt/subword_nmt/apply_bpe.py -c ${SRC_BPE_CODE} < ${dumpdir}/${ts}/${name}.tkn.tc.en > ${dumpdir}/${ts}/${name}.tkn.tc.en_bpe16000
    python subword-nmt/subword_nmt/apply_bpe.py -c ${TRG_BPE_CODE} < ${dumpdir}/${ts}/${name}.tkn.tc.${tgt_lang} > ${dumpdir}/${ts}/${name}.tkn.tc.${tgt_lang}_bpe16000
done
