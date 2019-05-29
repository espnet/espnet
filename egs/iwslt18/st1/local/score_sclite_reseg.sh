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
local/reorder_text.py data/${set}.en/text_noseg ${src}/FILE_ORDER > ${dir}/ref.wrd.trn || exit 1;
# local/reorder_text.py data/et_iwslt18_${set}.en/text_noseg ${src}/FILE_ORDER > ${dir}/ref.wrd.trn || exit 1;

# lowercasing
lowercase.perl < ${dir}/hyp.trn > ${dir}/hyp.trn.tmp
lowercase.perl < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.tmp
mv ${dir}/hyp.trn.tmp ${dir}/hyp.trn
mv ${dir}/ref.wrd.trn.tmp ${dir}/ref.wrd.trn

# remove punctuation
cat ${dir}/hyp.trn | local/remove_punctuation.pl | sed -e "s/  / /g" > ${dir}/hyp.trn.tmp
mv ${dir}/hyp.trn.tmp ${dir}/hyp.trn
cat ${dir}/ref.wrd.trn | local/remove_punctuation.pl | sed -e "s/  / /g" > ${dir}/ref.wrd.trn.tmp
mv ${dir}/ref.wrd.trn.tmp ${dir}/ref.wrd.trn

if [ ! -z ${bpemodel} ]; then
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
else
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
fi

# detokenize
cut -d " " -f 2- ${dir}/ref.wrd.trn | detokenizer.perl -l en -q > ${dir}/ref.wrd.trn.detok
cat ${dir}/hyp.wrd.trn | detokenizer.perl -l en -q > ${dir}/hyp.wrd.trn.detok
# NOTE: uttterance IDs are dummy

cat ${dir}/ref.wrd.trn.detok | perl local/wrap-xml.perl en ${xml_src} ${system} > ${dir}/ref.xml

# segment hypotheses with RWTH tool
# segmentBasedOnMWER.sh ${xml_src} ${xml_src} ${dir}/hyp.wrd.trn.detok ${system} en ${dir}/hyp.wrd.trn.detok.sgm.xml "" 0 || exit 1;
segmentBasedOnMWER.sh ${dir}/ref.xml ${dir}/ref.xml ${dir}/hyp.wrd.trn.detok ${system} en ${dir}/hyp.wrd.trn.detok.sgm.xml "" 0 || exit 1;
sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.detok.sgm.xml | awk '{print $0 "(uttID-"NR")"}' > ${dir}/hyp.wrd.trn.detok.sgm
cat ${dir}/ref.wrd.trn.detok | awk '{print $0 "(uttID-"NR")"}' > ${dir}/ref.wrd.trn.detok.tmp
mv ${dir}/ref.wrd.trn.detok.tmp ${dir}/ref.wrd.trn.detok
sclite -r ${dir}/ref.wrd.trn.detok trn -h ${dir}/hyp.wrd.trn.detok.sgm trn -i rm -o all stdout > ${dir}/result.wrd.txt

echo "write a WER result in ${dir}/result.wrd.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
