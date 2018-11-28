#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
word=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
lc=false

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2text.py ${dir}/data.json ${dic} ${dir}/ref ${dir}/hyp

if ${remove_blank}; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref ${dir}/ref.org
    cp ${dir}/hyp ${dir}/hyp.org
    filt.py -v ${nlsyms} ${dir}/ref.org > ${dir}/ref
    filt.py -v ${nlsyms} ${dir}/hyp.org > ${dir}/hyp
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp
    sed -i.bak3 -f ${filter} ${dir}/ref
fi

if ${lc}; then
  # case-insensitive
  multi-bleu.perl -lc ${dir}/ref < ${dir}/hyp > ${dir}/result.txt
else
  # case-sensitive
  multi-bleu.perl ${dir}/ref < ${dir}/hyp > ${dir}/result.txt
fi

echo "write a character-level BLEU result in ${dir}/result.txt"
cat ${dir}/result.txt

if ${word}; then
    if [ ! -z ${bpe} ]; then
    	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref | sed -e "s/▁/ /g" > ${dir}/ref.wrd
    	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp | sed -e "s/▁/ /g" > ${dir}/hyp.wrd
    else
      sed -e "s/ //g" -e "s/<space>/ /g" ${dir}/ref > ${dir}/ref.wrd
    	sed -e "s/ //g" -e "s/<space>/ /g" ${dir}/hyp > ${dir}/hyp.wrd
    fi

    if ${lc}; then
      # case-insensitive
      multi-bleu.perl -lc ${dir}/ref.wrd < ${dir}/hyp.wrd >> ${dir}/result.wrd.txt
    else
      # case-sensitive
      multi-bleu.perl ${dir}/ref.wrd < ${dir}/hyp.wrd > ${dir}/result.wrd.txt
    fi

    echo "write a word-level BLUE result in ${dir}/result.wrd.txt"
    cat ${dir}/result.wrd.txt
fi
