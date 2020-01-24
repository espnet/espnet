#! /bin/sh

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

SRC_VOCAB=${dumpdir}/vocab/vocab.en
TRG_VOCAB=${dumpdir}/vocab/vocab.${tgt_lang}

# train
local/generate_json.py  -s ${dumpdir}/${train_set}/train.tkn.tc.clean.en_bpe16000 -t ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang}_bpe16000 -sv ${SRC_VOCAB} -tv ${TRG_VOCAB} --dest ${dumpdir}/${train_set}/data.json

# valid
local/generate_json.py  -s ${dumpdir}/${train_dev}/tst2012.tkn.tc.en_bpe16000 -t ${dumpdir}/${train_dev}/tst2012.tkn.tc.${tgt_lang}_bpe16000 -sv ${SRC_VOCAB} -tv ${TRG_VOCAB} --dest ${dumpdir}/${train_dev}/data.json

# test
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    local/generate_json.py -s ${dumpdir}/${ts}/${name}.tkn.tc.en_bpe16000 -t ${dumpdir}/${ts}/${name}.tkn.tc.${tgt_lang}_bpe16000 -sv ${SRC_VOCAB} -tv ${TRG_VOCAB} --dest ${dumpdir}/${ts}/data.json
done
