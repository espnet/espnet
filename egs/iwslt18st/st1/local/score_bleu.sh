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

if [ $# != 4 ]; then
    echo "Usage: $0 <decode-dir> <dict> <data-dir> <set>";
    exit 1;
fi

dir=$1
dic=$2
set=$4
src=$3/$set/IWSLT.$set
system=st


# sort hypotheses
concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn_reorder.py ${dir}/data.json ${dic} ${dir}/hyp.trn $src/FILE_ORDER

if $remove_blank; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
fi

# reference
xml_en=$src/IWSLT.TED.$set.en-de.en.xml
xml_de=$src/IWSLT.TED.$set.en-de.de.xml

grep "<seg id" $xml_de | sed -e "s/<[^>]*>//g" > ${dir}/ref.trn # TODO(hirofumi):
grep "<seg id" $xml_de | sed -e "s/<[^>]*>//g" > ${dir}/ref.wrd.trn


### character-level
if [ -z $bpe ]; then
  # case-insensitive
  if ${lc}; then
    # segment hypotheses with RWTH tool
    segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.trn.detok $system de ${dir}/hyp.no-case.trn.detok.sgm.xml "" 0
    sed -e "/<[^>]*>/d" ${dir}/hyp.no-case.trn.detok.sgm.xml > ${dir}/hyp.no-case.trn.detok.sgm

    multi-bleu.perl -lc ${dir}/ref.trn < ${dir}/hyp.trn > ${dir}/result.no-case.txt
    echo "write a character-level BLEU result in ${dir}/result.no-case.txt"
    cat ${dir}/result.no-case.txt

  # case-sensitive
  else
    # segment hypotheses with RWTH tool
    segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.trn.detok $system de ${dir}/hyp.no-case.trn.detok.sgm.xml "" 0
    sed -e "/<[^>]*>/d" ${dir}/hyp.no-case.trn.detok.sgm.xml > ${dir}/hyp.no-case.trn.detok.sgm

    multi-bleu.perl ${dir}/ref.trn < ${dir}/hyp.trn > ${dir}/result.txt
    echo "write a character-level BLEU result in ${dir}/result.txt"
    cat ${dir}/result.txt
  fi
fi


### BPE-level
if [ ! -z $bpe ]; then
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
else
	sed -e "s/ //g" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
fi

# detokenize
detokenizer.perl -u -l de < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
# NOTE: uppercase the first character (-u)

# case-insensitive
if ${lc}; then
  # segment hypotheses with RWTH tool
  segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.wrd.trn.detok $system de ${dir}/hyp.no-case.wrd.trn.detok.sgm.xml "" 0
  sed -e "/<[^>]*>/d" ${dir}/hyp.no-case.wrd.trn.detok.sgm.xml > ${dir}/hyp.no-case.wrd.trn.detok.sgm

  multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn > ${dir}/result.no-case.wrd.txt
  echo "write a word-level BLUE result in ${dir}/result.no-case.wrd.txt"
  cat ${dir}/result.no-case.wrd.txt

# case-sensitive
else
  # segment hypotheses with RWTH tool
  segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.wrd.trn.detok $system de ${dir}/hyp.wrd.trn.detok.sgm.xml "" 1
  sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.detok.sgm.xml > ${dir}/hyp.wrd.trn.detok.sgm

  multi-bleu-detok.perl ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn.detok.sgm > ${dir}/result.wrd.txt
  echo "write a word-level BLUE result in ${dir}/result.wrd.txt"
  cat ${dir}/result.wrd.txt
fi

# TODO(hirofumi): add TER & METEOR metrics here
