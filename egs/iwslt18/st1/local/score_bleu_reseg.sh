#!/usr/bin/env bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpemodel=""
filter=""
case=tc
text=""
remove_nonverbal=true

. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: $0 <decode-dir> <dict> <data-dir> <set>";
    exit 1;
fi

dir=$1
dict=$2
set=$4
src=$3/${set}/IWSLT.${set}

src_lang=en
tgt_lang=de
sysid=st
xml_src=${src}/IWSLT.TED.${set}.${src_lang}-${tgt_lang}.${src_lang}.xml
xml_tgt=${src}/IWSLT.TED.${set}.${src_lang}-${tgt_lang}.${tgt_lang}.xml

# sort hypotheses
concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn_reorder.py ${dir}/data.json ${dict} ${dir}/hyp.trn.org ${src}/FILE_ORDER

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn

# remove non-verbal labels (optional)
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn > ${dir}/hyp.rm.trn

# if [ -n "${nlsyms}" ]; then
#     cp ${dir}/hyp.trn ${dir}/hyp.trn.org
#     filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
# fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
fi

# reorder text based on the order of the xml file
# if [ ! -n "${text}" ]; then
#   text=data/${set}.en/text_noseg.${case}
# fi
# local/reorder_text.py ${text} ${src}/FILE_ORDER > ${dir}/ref.wrd.trn || exit 1;
grep "<seg id" ${xml_tgt} | sed -e "s/<[^>]*>//g" | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' > ${dir}/ref.wrd.trn

if [ -n "${bpemodel}" ]; then
    if [ ${remove_nonverbal} = true ]; then
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.rm.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    else
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    fi
else
    if [ ${remove_nonverbal} = true ]; then
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/hyp.rm.trn > ${dir}/hyp.wrd.trn
    else
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    fi
fi

# detokenize
detokenizer.perl -lu de -q < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
detokenizer.perl -lu de -q < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
# NOTE: uppercase the first character (-u)

# remove language IDs
if [ -n "${nlsyms}" ]; then
    cp ${dir}/hyp.wrd.trn.detok ${dir}/hyp.wrd.trn.detok.tmp
    filt.py -v ${nlsyms} ${dir}/hyp.wrd.trn.detok.tmp > ${dir}/hyp.wrd.trn.detok
fi
# NOTE: this must be performed after detokenization so that punctuation marks are not removed

echo ${set} > ${dir}/result.${case}.txt
echo "########################################################################################################################" >> ${dir}/result.${case}.txt
echo "sacleBLEU" >> ${dir}/result.${case}.txt
if [ ${case} = tc ]; then
    ### case-sensitive
    # resegment hypotheses based on WER
    segmentBasedOnMWER.sh ${xml_src} ${xml_tgt} ${dir}/hyp.wrd.trn.detok ${sysid} ${tgt_lang} ${dir}/hyp.wrd.trn.detok.reseg.xml "" 1 || exit 1;
    sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.detok.reseg.xml > ${dir}/hyp.wrd.trn.detok.reseg
    sacrebleu ${dir}/ref.wrd.trn.detok -i ${dir}/hyp.wrd.trn.detok.reseg -m bleu chrf ter >> ${dir}/result.${case}.txt
else
    ### case-insensitive
    # resegment hypotheses based on WER
    segmentBasedOnMWER.sh  ${xml_src}  ${xml_tgt} ${dir}/hyp.wrd.trn.detok ${sysid} ${tgt_lang} ${dir}/hyp.wrd.trn.detok.reseg.xml "" 0 || exit 1;
    sed -e "/<[^>]*>/d" ${dir}/hyp.wrd.trn.detok.reseg.xml > ${dir}/hyp.wrd.trn.detok.reseg
    sacrebleu -lc ${dir}/ref.wrd.trn.detok -i ${dir}/hyp.wrd.trn.detok.reseg -m bleu chrf ter >> ${dir}/result.${case}.txt
fi
echo "write a case-insensitive BLEU result in ${dir}/result.${case}.txt"
echo "########################################################################################################################" >> ${dir}/result.${case}.txt
cat ${dir}/result.${case}.txt

# TODO(hirofumi): add METEOR, BERTscore here
