#!/usr/bin/env bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpemodel=""
filter=""
case=tc
set=""
remove_nonverbal=true
character_level=false

. utils/parse_options.sh

if [ $# -lt 3 ]; then
    echo "Usage: $0 <decode-dir> <tgt_lang> <dict-tgt> <dict-src>";
    exit 1;
fi

dir=$1
tgt_lang=$2
dic_tgt=$3
dic_src=$4

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn_mt.py ${dir}/data.json ${dic_tgt} --refs ${dir}/ref.trn.org \
    --hyps ${dir}/hyp.trn.org --srcs ${dir}/src.trn.org --dict-src ${dic_src}

# remove uttterance id
perl -pe 's/\([^\)]+\)\n/\n/g;' ${dir}/ref.trn.org > ${dir}/ref.trn
perl -pe 's/\([^\)]+\)\n/\n/g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn
perl -pe 's/\([^\)]+\)\n/\n/g;' ${dir}/src.trn.org > ${dir}/src.trn
perl -pe 's/.+\s\(([^\)]+)\)\n/\($1\)\n/g;' ${dir}/ref.trn.org > ${dir}/utt_id

# remove non-verbal labels (optional)
perl -pe 's/\([^\)]+\)//g;' ${dir}/ref.trn > ${dir}/ref.rm.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn > ${dir}/hyp.rm.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/src.trn > ${dir}/src.rm.trn

if [ -n "${bpemodel}" ]; then
    if [ ${remove_nonverbal} = true ]; then
        cat ${dir}/ref.rm.trn > ${dir}/ref.wrd.trn
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.rm.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/src.rm.trn | sed -e "s/▁/ /g" > ${dir}/src.wrd.trn
    else
        cat ${dir}/ref.trn > ${dir}/ref.wrd.trn
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/src.trn | sed -e "s/▁/ /g" > ${dir}/src.wrd.trn
    fi
else
    if [ ${remove_nonverbal} = true ]; then
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref.rm.trn > ${dir}/ref.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/hyp.rm.trn > ${dir}/hyp.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/src.rm.trn > ${dir}/src.wrd.trn
    else
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/src.trn > ${dir}/src.wrd.trn
    fi
fi

# detokenize
detokenizer.perl -l ${tgt_lang} -q < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
detokenizer.perl -l ${tgt_lang} -q < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
detokenizer.perl -l ${tgt_lang} -q < ${dir}/src.wrd.trn > ${dir}/src.wrd.trn.detok

# remove language IDs
if [ -n "${nlsyms}" ]; then
    cp ${dir}/ref.wrd.trn.detok ${dir}/ref.wrd.trn.detok.tmp
    cp ${dir}/hyp.wrd.trn.detok ${dir}/hyp.wrd.trn.detok.tmp
    cp ${dir}/src.wrd.trn.detok ${dir}/src.wrd.trn.detok.tmp
    filt.py -v $nlsyms ${dir}/ref.wrd.trn.detok.tmp > ${dir}/ref.wrd.trn.detok
    filt.py -v $nlsyms ${dir}/hyp.wrd.trn.detok.tmp > ${dir}/hyp.wrd.trn.detok
    filt.py -v $nlsyms ${dir}/src.wrd.trn.detok.tmp > ${dir}/src.wrd.trn.detok
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.wrd.trn.detok
    sed -i.bak3 -f ${filter} ${dir}/ref.wrd.trn.detok
    sed -i.bak3 -f ${filter} ${dir}/src.wrd.trn.detok
fi
# NOTE: this must be performed after detokenization so that punctuation marks are not removed

if [ ${character_level} = true ]; then
    # for Japanese/Chinese
    cp ${dir}/ref.wrd.trn.detok ${dir}/ref.wrd.trn.detok.tmp
    cp ${dir}/hyp.wrd.trn.detok ${dir}/hyp.wrd.trn.detok.tmp
    cp ${dir}/src.wrd.trn.detok ${dir}/src.wrd.trn.detok.tmp
    LC_ALL=en_US.UTF-8 sed -e 's/\(.\)/ \1/g' ${dir}/ref.wrd.trn.detok.tmp > ${dir}/ref.wrd.trn.detok
    LC_ALL=en_US.UTF-8 sed -e 's/\(.\)/ \1/g' ${dir}/hyp.wrd.trn.detok.tmp > ${dir}/hyp.wrd.trn.detok
    LC_ALL=en_US.UTF-8 sed -e 's/\(.\)/ \1/g' ${dir}/src.wrd.trn.detok.tmp > ${dir}/src.wrd.trn.detok
fi

if [ -f ${dir}/result.${case}.txt ]; then
    rm ${dir}/result.${case}.txt
    touch ${dir}/result.${case}.txt
fi
if [ -n "${set}" ]; then
    echo ${set} > ${dir}/result.${case}.txt
fi
echo "########################################################################################################################" >> ${dir}/result.${case}.txt
echo "sacleBLEU" >> ${dir}/result.${case}.txt
if [ ${case} = tc ]; then
    sacrebleu ${dir}/ref.wrd.trn.detok -i ${dir}/hyp.wrd.trn.detok -m bleu chrf ter >> ${dir}/result.${case}.txt
    echo "write a case-sensitive BLEU result in ${dir}/result.tc.txt"
else
    sacrebleu -lc ${dir}/ref.wrd.trn.detok -i ${dir}/hyp.wrd.trn.detok -m bleu chrf ter >> ${dir}/result.${case}.txt
    echo "write a case-insensitive BLEU result in ${dir}/result.lc.txt"
fi
echo "########################################################################################################################" >> ${dir}/result.${case}.txt
cat ${dir}/result.${case}.txt

# TODO(hirofumi): add METEOR, BERTscore here
