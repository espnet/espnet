#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
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
system=asr


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

# generate reference
xml_en=$src/IWSLT.TED.$set.en-de.en.xml

# normalize reference
grep "<seg id" $xml_en | sed -e "s/<[^>]*>//g" -e "s/[\.?,:\!]//g" -e 's/"//g' -e "s/-//g" | normalize-punctuation.perl -l en | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' | awk '{print $0 "(uttID-"NR")"}' > ${dir}/ref.wrd.trn

# NOTE: evaluate only with WER
if [ ! -z $bpe ]; then
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
else
	sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
fi

# detokenize
detokenizer.perl -u -l de < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
# NOTE: uppercase the first character (-u)

if ${lc}; then
  # segment hypotheses with RWTH tool
  segmentBasedOnMWER.sh $xml_en $xml_en ${dir}/hyp.wrd.trn.detok $system en ${dir}/hyp.no-case.wrd.trn.detok.sgm.xml "" 0
  sed -e "/<[^>]*>/d" ${dir}/hyp.no-case.wrd.trn.detok.sgm.xml | sed -e "s/[\.?,:\!]//g" -e 's/"//g' -e "s/-//g" | awk '{print $0 "(uttID-"NR")"}' > ${dir}/hyp.no-case.wrd.trn.detok.sgm

  awk '{ print tolower($0) }' < ${dir}/ref.wrd.trn > ${dir}/ref.no-case.wrd.trn
  sclite -r ${dir}/ref.no-case.wrd.trn trn -h ${dir}/hyp.no-case.wrd.trn.detok.sgm trn -i rm -o all stdout > ${dir}/result.no-case.wrd.txt

  echo "write a WER result in ${dir}/result.no-case.wrd.txt"
  grep -e Avg -e SPKR -m 2 ${dir}/result.no-case.wrd.txt
else
  # segment hypotheses with RWTH tool
  segmentBasedOnMWER.sh $xml_en $xml_en ${dir}/hyp.wrd.trn.detok $system en ${dir}/hyp.wrd.trn.detok.sgm.xml "" 1
  sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.detok.sgm.xml | sed -e "s/[\.?,:\!]//g" -e 's/"//g' -e "s/-//g" | awk '{print $0 "(uttID-"NR")"}' > ${dir}/hyp.wrd.trn.detok.sgm

  sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn.detok.sgm trn -i rm -o all stdout > ${dir}/result.wrd.txt

  echo "write a WER result in ${dir}/result.wrd.txt"
  grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
