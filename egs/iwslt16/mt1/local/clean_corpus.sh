#!/usr/bin/env bash

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1

data=$1
src_lang=$2
tgt_lang=$3
dumpdir=$4
train_set=$5


clean-corpus-n.perl ${dumpdir}/${train_set}/train.tkn.tc ${src_lang} ${tgt_lang} ${dumpdir}/${train_set}/train.tkn.tc.clean 1 50
