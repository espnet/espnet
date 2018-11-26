#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
filter=""

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <decode-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn.py ${dir}/data.json ${dic} ${dir}/ref.trn.org ${dir}/hyp.trn.org ${dir}/src.trn.org

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

spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
spm_decode --model=${bpemodel} --input_format=piece < ${dir}/src.trn | sed -e "s/▁/ /g" > ${dir}/src.wrd.trn

# detokenize
detokenizer.perl -l fr < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
detokenizer.perl -l fr < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
detokenizer.perl -l fr < ${dir}/src.wrd.trn > ${dir}/src.wrd.trn.detok

### case-insensitive
multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn.detok < ${dir}/hyp.wrd.trn.detok > ${dir}/result.wrd.txt
echo "write a case-insensitive word-level BLEU result in ${dir}/result.wrd.txt"
cat ${dir}/result.wrd.txt


# TODO(hirofumi): add TER & METEOR metrics here
