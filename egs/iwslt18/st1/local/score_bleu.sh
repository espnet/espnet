#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
filter=""
lc=false

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <decode-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn.py ${dir}/data.json ${dic} --ref ${dir}/ref.trn.org --hyp ${dir}/hyp.trn.org --src ${dir}/src.trn.org

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/ref.trn.org > ${dir}/ref.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/src.trn.org > ${dir}/src.trn

if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref.trn ${dir}/ref.trn.org
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    cp ${dir}/src.trn ${dir}/src.trn.org
    filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
    filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
    filt.py -v $nlsyms ${dir}/src.trn.org > ${dir}/src.trn
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
    sed -i.bak3 -f ${filter} ${dir}/ref.trn
    sed -i.bak3 -f ${filter} ${dir}/src.trn
fi

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
detokenizer.perl -l de < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
detokenizer.perl -l de < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
detokenizer.perl -l de < ${dir}/src.wrd.trn > ${dir}/src.wrd.trn.detok

sleep 1

if ${lc}; then
    # case-insensitive
    multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn.detok < ${dir}/hyp.wrd.trn.detok > ${dir}/result.txt
    echo "write a case-insensitive BLEU result in ${dir}/result.txt"
else
    # case-sensitive
    multi-bleu-detok.perl ${dir}/ref.wrd.trn.detok < ${dir}/hyp.wrd.trn.detok > ${dir}/result.txt
    echo "write a case-sensitive BLEU result in ${dir}/result.txt"
fi
cat ${dir}/result.txt


# TODO(hirofumi): add TER & METEOR metrics here
