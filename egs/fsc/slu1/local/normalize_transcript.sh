#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

data=$1
lang=en

cut -d' ' -f1 ${data}/text > ${data}/utt
cut -d' ' -f2- ${data}/text > ${data}/text_only

normalize-punctuation.perl -l ${lang} < ${data}/text_only > ${data}/text.norm

# lowercasing
lowercase.perl < ${data}/text.norm > ${data}/text.norm.lc
#cp ${data}/text.norm ${data}/text.norm.tc

# remove punctuation
local/remove_punctuation.pl < ${data}/text.norm.lc > ${data}/text.norm.lc.rm

#tokenization
#tokenizer.perl -l ${lang} -q < ${data}/text.norm.tc > ${data}/text.norm.tc.tok
#tokenizer.perl -l ${lang} -q < ${data}/text.norm.lc > ${data}/text.norm.lc.tok
#tokenizer.perl -l ${lang} -q < ${data}/text.norm.lc.rm > ${data}/text.norm.lc.rm.tok

paste -d " " ${data}/utt ${data}/text.norm.lc.rm > ${data}/text.lc.rm
