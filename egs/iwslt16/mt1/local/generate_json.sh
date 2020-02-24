#! /bin/sh

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

SRC_VOCAB=${dumpdir}/vocab/vocab.${src_lang}
TRG_VOCAB=${dumpdir}/vocab/vocab.${tgt_lang}

# train
local/generate_json.py  -s ${dumpdir}/${train_set}/train.tkn.tc.clean.${src_lang}_bpe${nbpe} -t ${dumpdir}/${train_set}/train.tkn.tc.clean.${tgt_lang}_bpe${nbpe} -sv ${SRC_VOCAB} -tv ${TRG_VOCAB} --dest ${dumpdir}/${train_set}/data.json

# valid
local/generate_json.py  -s ${dumpdir}/${train_dev}/tst2012.tkn.tc.${src_lang}_bpe${nbpe} -t ${dumpdir}/${train_dev}/tst2012.tkn.tc.${tgt_lang}_bpe${nbpe} -sv ${SRC_VOCAB} -tv ${TRG_VOCAB} --dest ${dumpdir}/${train_dev}/data.json

# test
for ts in $(echo ${trans_set} | tr '_' ' '); do
    name=`echo $ts | cut -d'.' -f1`
    local/generate_json.py -s ${dumpdir}/${ts}/${name}.tkn.tc.${src_lang}_bpe${nbpe} -t ${dumpdir}/${ts}/${name}.tkn.tc.${tgt_lang}_bpe${nbpe} -sv ${SRC_VOCAB} -tv ${TRG_VOCAB} --dest ${dumpdir}/${ts}/data.json
done
