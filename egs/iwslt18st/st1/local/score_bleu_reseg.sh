#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
remove_blank=true
filter=""

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
local/json2trn_reorder.py ${dir}/data.json ${dic} ${dir}/hyp.de.trn $src/FILE_ORDER

if $remove_blank; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.de.trn
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/hyp.de.trn ${dir}/hyp.de.trn.org
    filt.py -v $nlsyms ${dir}/hyp.de.trn.org > ${dir}/hyp.de.trn
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.de.trn
fi

# generate reference
xml_en=$src/IWSLT.TED.$set.en-de.en.xml
xml_de=$src/IWSLT.TED.$set.en-de.de.xml

grep "<seg id" $xml_de | sed -e "s/<[^>]*>//g" | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' > ${dir}/ref.de.trn
grep "<seg id" $xml_de | sed -e "s/<[^>]*>//g" | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' > ${dir}/ref.de.wrd.trn


### character-level
if [ -z $bpe ]; then
  # detokenize
  detokenizer.perl -l de < ${dir}/hyp.de.trn > ${dir}/hyp.de.trn.detok

  ### case-insensitive
  # segment hypotheses with RWTH tool
  segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.de.trn.detok $system de ${dir}/hyp.de.no-case.trn.detok.sgm.xml "" 0
  sed -e "/<[^>]*>/d" ${dir}/hyp.de.no-case.trn.detok.sgm.xml > ${dir}/hyp.de.no-case.trn.detok.sgm

  multi-bleu.perl -lc ${dir}/ref.de.trn < ${dir}/hyp.de.trn > ${dir}/result.no-case.txt
  echo "write a character-level case-insensitive BLEU result in ${dir}/result.no-case.txt"
  cat ${dir}/result.no-case.txt

  ### case-sensitive
  # segment hypotheses with RWTH tool
  segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.de.trn.detok $system de ${dir}/hyp.de.no-case.trn.detok.sgm.xml "" 0
  sed -e "/<[^>]*>/d" ${dir}/hyp.de.no-case.trn.detok.sgm.xml > ${dir}/hyp.de.no-case.trn.detok.sgm

  multi-bleu.perl ${dir}/ref.de.trn < ${dir}/hyp.de.trn > ${dir}/result.txt
  echo "write a character-level case-sensitive BLEU result in ${dir}/result.txt"
  cat ${dir}/result.txt
fi


### BPE-level
if [ ! -z $bpe ]; then
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.de.trn | sed -e "s/▁/ /g" > ${dir}/hyp.de.wrd.trn
else
	sed -e "s/ //g" -e "s/<space>/ /g" ${dir}/hyp.de.trn > ${dir}/hyp.de.wrd.trn
fi

# detokenize
detokenizer.perl -u -l de < ${dir}/hyp.de.wrd.trn > ${dir}/hyp.de.wrd.trn.detok
# NOTE: uppercase the first character (-u)

### case-insensitive
# segment hypotheses with RWTH tool
segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.de.wrd.trn.detok $system de ${dir}/hyp.de.no-case.wrd.trn.detok.sgm.xml "" 0
sed -e "/<[^>]*>/d" ${dir}/hyp.de.no-case.wrd.trn.detok.sgm.xml > ${dir}/hyp.de.no-case.wrd.trn.detok.sgm

multi-bleu-detok.perl -lc ${dir}/ref.de.wrd.trn < ${dir}/hyp.de.no-case.wrd.trn.detok.sgm > ${dir}/result.no-case.wrd.txt
echo "write a case-insensitive word-level BLUE result in ${dir}/result.no-case.wrd.txt"
cat ${dir}/result.no-case.wrd.txt

### case-sensitive
# segment hypotheses with RWTH tool
segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.de.wrd.trn.detok $system de ${dir}/hyp.de.wrd.trn.detok.sgm.xml "" 1
sed -e "/<[^>]*>/d" ${dir}/hyp.de.wrd.trn.detok.sgm.xml > ${dir}/hyp.de.wrd.trn.detok.sgm

multi-bleu-detok.perl ${dir}/ref.de.wrd.trn < ${dir}/hyp.de.wrd.trn.detok.sgm > ${dir}/result.wrd.txt
echo "write a case-sensitve word-level BLUE result in ${dir}/result.wrd.txt"
cat ${dir}/result.wrd.txt


# TODO(hirofumi): add TER & METEOR metrics here
