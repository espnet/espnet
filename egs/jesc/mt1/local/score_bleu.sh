#!/usr/bin/env bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# TL;DR:
# To score blue, the files hyp.trn and ref.trn are used instead of *.wrd.trn or *.detok
# This is because the translation add some character in jpn and then it is scored as 0
# as example where ref: ああ 約束するぜ and hyp: ええ 約束する
# detok deletes the space to ああ 約束するぜ becomes ああ約束するぜ and ええ約束する which are
# complete different words and when using *.wrd you see that 約束するぜ is different
# from 約束する by only one char (and does not change that much the meaning, you could
# ask @sw005320 about it ;)
# So then to evaluate it, hyp/ref.trn are used (e.g.); ref: あ あ <space> 約 束 す る ぜ
# and hyp: え え <space> 約 束 す る.

export LC_ALL=C

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
filter=""
case=lc
set=""

. utils/parse_options.sh

if [ $# -lt 3 ]; then
    echo "Usage: $0 <decode-dir> <tgt_lang> <dict-tgt> <dict-src>";
    exit 1;
fi

dir=$1
tgt_lang=$2
dic_tgt=$3
dic_src=$4

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn_mt.py ${dir}/data.json ${dic_tgt} --refs ${dir}/ref.trn.org \
    --hyps ${dir}/hyp.trn.org --srcs ${dir}/src.trn.org --dict-src ${dic_src}

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/ref.trn.org > ${dir}/ref.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/src.trn.org > ${dir}/src.trn

if [ ${case} = tc ]; then
    echo ${set} > ${dir}/result.tc.txt
    multi-bleu-detok.perl ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn >> ${dir}/result.tc.txt
    echo "write a case-sensitive BLEU result in ${dir}/result.tc.txt"
    cat ${dir}/result.tc.txt
fi

echo ${set} > ${dir}/result.lc.txt
multi-bleu-detok.perl -lc ${dir}/ref.trn < ${dir}/hyp.trn > ${dir}/result.lc.txt
echo "write a case-insensitive BLEU result in ${dir}/result.lc.txt"
cat ${dir}/result.lc.txt

# TODO(hirofumi): add TER & METEOR metrics here
