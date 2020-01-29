#!/bin/bash

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1

data=$1
tgt_lang=$2
dumpdir=$3
train_set=$4


clean-corpus-n.perl ${dumpdir}/${train_set}/train.tkn.tc en ${tgt_lang} ${dumpdir}/${train_set}/train.tkn.tc.clean 1 50