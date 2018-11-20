#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
filter=""
set=""

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <decode-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn.py ${dir}/data.json ${dic} ${dir}/ref.trn.org ${dir}/hyp.trn.org ${dir}/src.trn.org
if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
  local/json2trn.py ${dir}/data_ref1.json ${dic} ${dir}/ref1.trn.org
  local/json2trn.py ${dir}/data_ref2.json ${dic} ${dir}/ref2.trn.org
  local/json2trn.py ${dir}/data_ref3.json ${dic} ${dir}/ref3.trn.org
fi

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/ref.trn.org > ${dir}/ref.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/src.trn.org > ${dir}/src.trn
if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
  perl -pe 's/\([^\)]+\)//g;' ${dir}/ref1.trn.org > ${dir}/ref1.trn
  perl -pe 's/\([^\)]+\)//g;' ${dir}/ref2.trn.org > ${dir}/ref2.trn
  perl -pe 's/\([^\)]+\)//g;' ${dir}/ref3.trn.org > ${dir}/ref3.trn
fi

if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref.trn ${dir}/ref.trn.org
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    cp ${dir}/src.trn ${dir}/src.trn.org
    filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
    filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
    filt.py -v $nlsyms ${dir}/src.trn.org > ${dir}/src.trn
    if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
      cp ${dir}/ref1.trn ${dir}/ref1.trn.org
      cp ${dir}/ref2.trn ${dir}/ref2.trn.org
      cp ${dir}/ref3.trn ${dir}/ref3.trn.org
      filt.py -v $nlsyms ${dir}/ref1.trn.org > ${dir}/ref1.trn
      filt.py -v $nlsyms ${dir}/ref2.trn.org > ${dir}/ref2.trn
      filt.py -v $nlsyms ${dir}/ref3.trn.org > ${dir}/ref3.trn
    fi
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
    sed -i.bak3 -f ${filter} ${dir}/ref.trn
    sed -i.bak3 -f ${filter} ${dir}/src.trn
    if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
      sed -i.bak3 -f ${filter} ${dir}/ref1.trn
      sed -i.bak3 -f ${filter} ${dir}/ref2.trn
      sed -i.bak3 -f ${filter} ${dir}/ref3.trn
    fi
fi

if [ ! -z ${bpemodel} ]; then
  spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
  spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
  spm_decode --model=${bpemodel} --input_format=piece < ${dir}/src.trn | sed -e "s/▁/ /g" > ${dir}/src.wrd.trn
  if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref1.trn | sed -e "s/▁/ /g" >> ${dir}/ref.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref2.trn | sed -e "s/▁/ /g" >> ${dir}/ref.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref3.trn | sed -e "s/▁/ /g" >> ${dir}/ref.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" >> ${dir}/hyp.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" >> ${dir}/hyp.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" >> ${dir}/hyp.wrd.trn
  fi
else
  sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
  sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
  sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/src.trn > ${dir}/src.wrd.trn
  if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref1.trn >> ${dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref2.trn >> ${dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref3.trn >> ${dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn >> ${dir}/hyp.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn >> ${dir}/hyp.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn >> ${dir}/hyp.wrd.trn
  fi
fi

# detokenize
detokenizer.perl -l en < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
detokenizer.perl -l en < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
detokenizer.perl -l en < ${dir}/src.wrd.trn > ${dir}/src.wrd.trn.detok

### case-insensitive
multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn.detok < ${dir}/hyp.wrd.trn.detok > ${dir}/result.wrd.txt
echo "write a case-insensitive word-level BLUE result in ${dir}/result.wrd.txt"
cat ${dir}/result.wrd.txt


# TODO(hirofumi): add TER & METEOR metrics here
