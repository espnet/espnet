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
nbpe=$8

if [ -e subword-nmt ]; then
  echo "subword-nmt toolkit already exists. continue..."
else
  echo "subword-nmt toolkit not found. cloning..."
  git clone https://github.com/rsennrich/subword-nmt
fi

mkdir -p ${dumpdir}/vocab

SRC_BPE_CODE=${dumpdir}/vocab/vocab.${src_lang}_bpe16000
TRG_BPE_CODE=${dumpdir}/vocab/vocab.${tgt_lang}_bpe16000

# learn bpe merge operation
echo "learn bpe merge operation"
python3 subword-nmt/subword_nmt/learn_bpe.py -s $nbpe < ${dumpdir}/${train_set}/train.tkn.tc.clean.${src_lang} > ${SRC_BPE_CODE}
python3 subword-nmt/subword_nmt/learn_bpe.py -s $nbpe < ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang} > ${TRG_BPE_CODE}

# apply bpe operation
echo "apply bpe splitting"
# train
python3 subword-nmt/subword_nmt/apply_bpe.py -c ${SRC_BPE_CODE} < ${dumpdir}/${train_set}/train.tkn.tc.clean.${src_lang} > ${dumpdir}/${train_set}/train.tkn.tc.clean.${src_lang}_bpe${nbpe}
python3 subword-nmt/subword_nmt/apply_bpe.py -c ${TRG_BPE_CODE} < ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang} > ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang}_bpe${nbpe}


# valid
python3 subword-nmt/subword_nmt/apply_bpe.py -c ${SRC_BPE_CODE} < ${dumpdir}/${train_dev}/tst2012.tkn.tc.${src_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.tc.${src_lang}_bpe${nbpe}
python3 subword-nmt/subword_nmt/apply_bpe.py -c ${TRG_BPE_CODE} < ${dumpdir}/${train_dev}/tst2012.tkn.tc.${tgt_lang} > ${dumpdir}/${train_dev}/tst2012.tkn.tc.${tgt_lang}_bpe${nbpe}

# test
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    python3 subword-nmt/subword_nmt/apply_bpe.py -c ${SRC_BPE_CODE} < ${dumpdir}/${ts}/${name}.tkn.tc.${src_lang} > ${dumpdir}/${ts}/${name}.tkn.tc.${src_lang}_bpe${nbpe}
    python3 subword-nmt/subword_nmt/apply_bpe.py -c ${TRG_BPE_CODE} < ${dumpdir}/${ts}/${name}.tkn.tc.${tgt_lang} > ${dumpdir}/${ts}/${name}.tkn.tc.${tgt_lang}_bpe${nbpe}
done
