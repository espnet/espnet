#!/usr/bin/env bash

# Copyright 2021 Johns Hopkins University (Jiatong Shi)
# Refactored from egs/must_c/st1/local
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


export LC_ALL=C

./utils/parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dst> <src-lang> <tgt-lang>"
    echo "e.g.: $0 /n/rd11/corpora_8/MUSTC_v1.0 es en"
    exit 1;
fi

dst=$1
src_lang=$2
tgt_lang=$3

for lang in ${src_lang} ${tgt_lang}; do
    # normalize punctuation
    normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org > ${dst}/${lang}.norm

    # lowercasing
    lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
    cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

    # remove punctuation
    remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

    # tokenization
    # no tokenizer for swa, swc, skip it
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok

    paste -d " " ${dst}/utt_id ${dst}/${lang}.norm.tc.tok | sort > ${dst}/text.tc.${lang}
    paste -d " " ${dst}/utt_id ${dst}/${lang}.norm.lc.tok | sort > ${dst}/text.lc.${lang}
    paste -d " " ${dst}/utt_id ${dst}/${lang}.norm.lc.rm.tok | sort > ${dst}/text.lc.rm.${lang}

    # save original and cleaned punctuation
    lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
    lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
done
