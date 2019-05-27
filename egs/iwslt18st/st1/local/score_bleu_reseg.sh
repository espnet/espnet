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

if [ $# != 4 ]; then
    echo "Usage: $0 <decode-dir> <dict> <data-dir> <set>";
    exit 1;
fi

dir=$1
dic=$2
set=$4
src=$3/$set/IWSLT.$set

sl=en
tl=de
system=st
xml_src=${src}/IWSLT.TED.${set}.${sl}-${tl}.${sl}.xml
xml_tgt=${src}/IWSLT.TED.${set}.${sl}-${tl}.${tl}.xml

# sort hypotheses
concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn_reorder.py ${dir}/data.json ${dic} ${dir}/hyp.trn.org ${src}/FILE_ORDER

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn

if [ -n "${nlsyms}" ]; then
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
fi

# local/reorder_text.py data/${set}.en/text_noseg ${src}/FILE_ORDER > ${dir}/src.wrd.trn || exit 1;
# local/reorder_text.py data/${set}.de/text_noseg ${src}/FILE_ORDER > ${dir}/ref.wrd.trn || exit 1;

grep "<seg id" ${xml_src} | sed -e "s/<[^>]*>//g" | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' > ${dir}/src.wrd.trn
grep "<seg id" ${xml_tgt} | sed -e "s/<[^>]*>//g" | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' > ${dir}/ref.wrd.trn

# normalize punctuation
# cat ${dir}/ref.wrd.trn | normalize-punctuation.perl -l de | sed -e "s/‒/ - /g" | sed -e "s/‟/\"/g" | sed -e "s/，/,/g" | \
#   local/normalize_punctuation.pl > ${dir}/ref.wrd.trn.tmp
# TODO(hirofumi): remove later
cat ${dir}/ref.wrd.trn | normalize-punctuation.perl -l de | sed -e "s/‒/ - /g" | sed -e "s/‟/\"/g" | sed -e "s/，/,/g" | sed -e "s/\.//g" | \
  local/normalize_punctuation.pl > ${dir}/ref.wrd.trn.tmp
cat ${dir}/src.wrd.trn | normalize-punctuation.perl -l de | sed -e "s/‒/ - /g" | sed -e "s/‟/\"/g" | sed -e "s/，/,/g" | \
  local/normalize_punctuation.pl > ${dir}/src.wrd.trn.tmp
mv ${dir}/src.wrd.trn.tmp ${dir}/src.wrd.trn
mv ${dir}/ref.wrd.trn.tmp ${dir}/ref.wrd.trn

# lowercase
if ${lc}; then
    lowercase.perl < ${dir}/src.wrd.trn > ${dir}/src.wrd.trn.tmp
    lowercase.perl < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.tmp
    mv ${dir}/src.wrd.trn.tmp ${dir}/src.wrd.trn
    mv ${dir}/ref.wrd.trn.tmp ${dir}/ref.wrd.trn
fi

# NOTE: these are used for segementation
lowercase.perl < ${xml_src} > ${dir}/src.xml
lowercase.perl < ${xml_tgt} > ${dir}/ref.xml

if [ ! -z ${bpemodel} ]; then
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
else
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
fi

# detokenize
detokenizer.perl -l de -q < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
# NOTE: uppercase the first character (-u)

# paste -d " " <(cut -d " " -f 1 ${dir}/src.wrd.trn) <(cat ${dir}/src.wrd.trn.detok) > ${dir}/text_noseg.en.reorder.detok
# paste -d " " <(cut -d " " -f 1 ${dir}/ref.wrd.trn) <(cat ${dir}/ref.wrd.trn.detok) > ${dir}/text_noseg.de.reorder.detok
# local/text2xml.py ${dir}/text_noseg.en.reorder.detok > ${dir}/xml.en || exit 1;
# local/text2xml.py ${dir}/text_noseg.de.reorder.detok > ${dir}/xml.de || exit 1;

# cat ${dir}/src.wrd.trn.detok | perl local/wrap-xml.perl german ${xml_src} ${system} > ${dir}/src.xml
# cat ${dir}/ref.wrd.trn.detok | perl local/wrap-xml.perl de ${xml_tgt} ${system} > ${dir}/ref.xml

# cp ${xml_src} ${dir}/src.org.xml
# cp ${xml_tgt} ${dir}/ref.org.xml

# cat $xml_src > ${dir}/xml.en.org
# cat $xml_tgt > ${dir}/xml.de.org

if ${lc}; then
  ### case-insensitive
  # segment hypotheses with RWTH tool
  # segmentBasedOnMWER.sh ${xml_src} ${dir}/xml.de ${dir}/hyp.wrd.trn.detok ${system} de ${dir}/hyp.wrd.trn.detok.sgm.xml "" 0 || exit 1;
  # segmentBasedOnMWER.sh ${dir}/xml.en.org ${dir}/xml.de.org ${dir}/hyp.wrd.trn.detok ${system} de ${dir}/hyp.wrd.trn.detok.sgm.xml "" 0 || exit 1;
  # segmentBasedOnMWER.sh ${dir}/src.xml ${dir}/hyp.xml ${dir}/hyp.wrd.trn.detok ${system} de ${dir}/hyp.wrd.trn.detok.sgm.xml "" 0 || exit 1;
  # segmentBasedOnMWER.sh ${xml_src} ${xml_tgt} ${dir}/hyp.wrd.trn.detok ${system} de ${dir}/hyp.wrd.trn.detok.sgm.xml "" 0 || exit 1;
  segmentBasedOnMWER.sh ${dir}/src.xml ${dir}/ref.xml ${dir}/hyp.wrd.trn.detok ${system} de ${dir}/hyp.wrd.trn.detok.sgm.xml "" 0 || exit 1;
  sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.detok.sgm.xml > ${dir}/hyp.wrd.trn.detok.sgm
  multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn.detok.sgm > ${dir}/result.txt

  echo "write a case-insensitive word-level BLUE result in ${dir}/result.txt"
else
  ### case-sensitive
  # segment hypotheses with RWTH tool
  # segmentBasedOnMWER.sh ${dir}/xml.en ${dir}/xml.de ${dir}/hyp.wrd.trn.detok ${system} de ${dir}/hyp.wrd.trn.detok.sgm.xml "" 1 || exit 1;
  # sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.detok.sgm.xml > ${dir}/hyp.wrd.trn.detok.sgm
  # multi-bleu-detok.perl ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn.detok.sgm > ${dir}/result.case.txt

  echo "write a case-insensitive BLEU result in ${dir}/result.txt"
fi
cat ${dir}/result.txt


# TODO(hirofumi): add TER & METEOR metrics here
