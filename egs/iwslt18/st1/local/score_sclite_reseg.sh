#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
case=lc.rm
text=""

. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: $0 <decode-dir> <dict> <data-dir> <set>";
    exit 1;
fi

dir=$1
dic=$2
set=$4
src=$3/${set}/IWSLT.${set}

sl=en
tl=de
system=asr
xml_src=${src}/IWSLT.TED.${set}.${sl}-${tl}.${sl}.xml

# sort hypotheses
concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn_reorder.py ${dir}/data.json ${dic} ${dir}/hyp.trn.org ${src}/FILE_ORDER

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn

if ${remove_blank}; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ -n "${nlsyms}" ]; then
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
fi

# reorder text based on the order of the xml file
if [ -z ${text} ]; then
  text=data/${set}.en/text_noseg.${case}
fi
local/reorder_text.py ${text} ${src}/FILE_ORDER > ${dir}/ref.wrd.trn || exit 1;
# grep "<seg id" ${xml_src} | sed -e "s/<[^>]*>//g" | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' > ${dir}/ref.wrd.trn

# lowercasing
lowercase.perl < ${dir}/hyp.trn > ${dir}/hyp.trn.lc
lowercase.perl < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.lc

# remove punctuation
local/remove_punctuation.pl < ${dir}/hyp.trn.lc | sed -e "s/  / /g" > ${dir}/hyp.trn.lc.rm
local/remove_punctuation.pl < ${dir}/ref.wrd.trn.lc | sed -e "s/  / /g" > ${dir}/ref.wrd.trn.lc.rm

if [ ! -z ${bpemodel} ]; then
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn.lc.rm | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn.lc.rm
else
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn.lc.rm > ${dir}/hyp.wrd.trn.lc.rm
fi

# detokenize
cut -d " " -f 2- ${dir}/ref.wrd.trn.lc.rm | detokenizer.perl -l en -q > ${dir}/ref.wrd.trn.lc.rm.detok
detokenizer.perl -l en -q < ${dir}/hyp.wrd.trn.lc.rm > ${dir}/hyp.wrd.trn.lc.rm.detok
# NOTE: uttterance IDs are dummy

perl local/wrap-xml.perl en ${xml_src} ${system} < ${dir}/ref.wrd.trn.lc.rm.detok > ${dir}/ref.xml

# segment hypotheses with RWTH tool
# segmentBasedOnMWER.sh ${xml_src} ${xml_src} ${dir}/hyp.wrd.trn.lc.rm.detok ${system} en ${dir}/hyp.wrd.trn.lc.rm.detok.sgm.xml "" 0 || exit 1;
segmentBasedOnMWER.sh ${dir}/ref.xml ${dir}/ref.xml ${dir}/hyp.wrd.trn.lc.rm.detok ${system} en ${dir}/hyp.wrd.trn.lc.rm.detok.sgm.xml "" 0 || exit 1;
sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.lc.rm.detok.sgm.xml | awk '{print $0 "(uttID-"NR")"}' > ${dir}/hyp.wrd.trn.lc.rm.detok.sgm
awk '{print $0 "(uttID-"NR")"}' < ${dir}/ref.wrd.trn.lc.rm.detok > ${dir}/ref.wrd.trn.lc.rm.detok.tmp
mv ${dir}/ref.wrd.trn.lc.rm.detok.tmp ${dir}/ref.wrd.trn.lc.rm.detok
sclite -r ${dir}/ref.wrd.trn.lc.rm.detok trn -h ${dir}/hyp.wrd.trn.lc.rm.detok.sgm trn -i rm -o all stdout > ${dir}/result.wrd.txt

echo "write a WER result in ${dir}/result.wrd.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
