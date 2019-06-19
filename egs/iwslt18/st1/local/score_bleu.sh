#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
filter=""
case=lc

. utils/parse_options.sh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <decode-dir> <dict-tgt> <dict-src>";
    exit 1;
fi

dir=$1
dic_tgt=$2
dic_src=$3

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn_mt.py ${dir}/data.json ${dic_tgt} --refs ${dir}/ref.trn.org \
    --hyps ${dir}/hyp.trn.org --srcs ${dir}/src.trn.org --dict-src ${dic_src}

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/ref.trn.org > ${dir}/ref.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/src.trn.org > ${dir}/src.trn

if [ ! -z ${bpemodel} ]; then
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/src.trn | sed -e "s/▁/ /g" > ${dir}/src.wrd.trn
else
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/src.trn > ${dir}/src.wrd.trn
fi

# detokenize
detokenizer.perl -l de -q < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
detokenizer.perl -l de -q < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
detokenizer.perl -l de -q < ${dir}/src.wrd.trn > ${dir}/src.wrd.trn.detok

if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref.wrd.trn.detok ${dir}/ref.wrd.trn.detok.org
    cp ${dir}/hyp.wrd.trn.detok ${dir}/hyp.wrd.trn.detok.org
    cp ${dir}/src.wrd.trn.detok ${dir}/src.wrd.trn.detok.org
    filt.py -v $nlsyms ${dir}/ref.wrd.trn.detok.org > ${dir}/ref.wrd.trn.detok
    filt.py -v $nlsyms ${dir}/hyp.wrd.trn.detok.org > ${dir}/hyp.wrd.trn.detok
    filt.py -v $nlsyms ${dir}/src.wrd.trn.detok.org > ${dir}/src.wrd.trn.detok
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.wrd.trn.detok
    sed -i.bak3 -f ${filter} ${dir}/ref.wrd.trn.detok
    sed -i.bak3 -f ${filter} ${dir}/src.wrd.trn.detok
fi

if [ ${case} = tc ]; then
    echo ${set} > ${dir}/result.tc.txt
    multi-bleu-detok.perl ${dir}/ref.wrd.trn.detok < ${dir}/hyp.wrd.trn.detok >> ${dir}/result.tc.txt
    echo "write a case-sensitive BLEU result in ${dir}/result.tc.txt"
    cat ${dir}/result.tc.txt
fi

# detokenize
local/remove_punctuation.pl < ${dir}/ref.wrd.trn.detok > ${dir}/ref.wrd.trn.detok.lc.rm
local/remove_punctuation.pl < ${dir}/hyp.wrd.trn.detok > ${dir}/hyp.wrd.trn.detok.lc.rm
local/remove_punctuation.pl < ${dir}/src.wrd.trn.detok > ${dir}/src.wrd.trn.detok.lc.rm

echo ${set} > ${dir}/result.lc.txt
multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn.detok.lc.rm < ${dir}/hyp.wrd.trn.detok.lc.rm >> ${dir}/result.lc.txt
echo "write a case-insensitive BLEU result in ${dir}/result.lc.txt"
cat ${dir}/result.lc.txt

# TODO(hirofumi): add TER & METEOR metrics here
