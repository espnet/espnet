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

# reorder text based on the order of the xml file
# if [ -z ${text} ]; then
#   text=data/${set}.en/text_noseg.${case}
# fi
# local/reorder_text.py ${text} ${src}/FILE_ORDER > ${dir}/ref.wrd.trn || exit 1;
grep "<seg id" ${xml_tgt} | sed -e "s/<[^>]*>//g" | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' > ${dir}/ref.wrd.trn

if [ ! -z ${bpemodel} ]; then
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
else
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
fi

# detokenize
detokenizer.perl -l de -q < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
# NOTE: uppercase the first character (-u)

if [ ${case} = tc ]; then
    ### case-sensitive
    exit 1  # TODO
    echo ${set} > ${dir}/result.tc.txt
    echo "write a case-sensitive BLEU result in ${dir}/result.tc.txt"
    cat ${dir}/result.tc.txt
fi

# lowercase
lowercase.perl < ${xml_src} > ${dir}/src.xml
lowercase.perl < ${xml_tgt} > ${dir}/ref.xml
# NOTE: these are used for segementation

### case-insensitive
echo ${set} > ${dir}/result.lc.txt
# segment hypotheses with RWTH tool
segmentBasedOnMWER.sh ${dir}/src.xml ${dir}/ref.xml ${dir}/hyp.wrd.trn.detok ${system} de ${dir}/hyp.wrd.trn.detok.sgm.xml "" 0 || exit 1;
sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.detok.sgm.xml > ${dir}/hyp.wrd.trn.detok.sgm
multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn.detok.sgm > ${dir}/result.txt
echo "write a case-insensitive word-level BLEU result in ${dir}/result.txt"
cat ${dir}/result.txt


# TODO(hirofumi): add TER & METEOR metrics here
