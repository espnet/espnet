#!/usr/bin/env bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpemodel=""
filter=""
case=lc
set=""

. utils/parse_options.sh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <decode-dir> <dict-tgt> <dict-src>";
    exit 1;
fi

dir=$1
dic_tgt=$2
dic_src=$3

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn_mt.py ${dir}/data.json ${dic_tgt} --refs ${dir}/ref.trn.org \
    --hyps ${dir}/hyp.trn.org --srcs ${dir}/src.trn.org --dict-src ${dic_src}
if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
    json2trn_mt.py ${dir}/data_ref1.json ${dic_tgt} --refs ${dir}/ref1.trn.org
    json2trn_mt.py ${dir}/data_ref2.json ${dic_tgt} --refs ${dir}/ref2.trn.org
    json2trn_mt.py ${dir}/data_ref3.json ${dic_tgt} --refs ${dir}/ref3.trn.org
fi

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/ref.trn.org > ${dir}/ref.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/src.trn.org > ${dir}/src.trn
if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
    perl -pe 's/\([^\)]+\)//g;' ${dir}/ref1.trn.org > ${dir}/ref1.trn
    perl -pe 's/\([^\)]+\)//g;' ${dir}/ref2.trn.org > ${dir}/ref2.trn
    perl -pe 's/\([^\)]+\)//g;' ${dir}/ref3.trn.org > ${dir}/ref3.trn
fi

if [ -n "${bpemodel}" ]; then
    cat ${dir}/ref.trn > ${dir}/ref.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/src.trn | sed -e "s/▁/ /g" > ${dir}/src.wrd.trn
    if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
        cat ${dir}/ref1.trn > ${dir}/ref1.wrd.trn
        cat ${dir}/ref2.trn > ${dir}/ref2.wrd.trn
        cat ${dir}/ref3.trn > ${dir}/ref3.wrd.trn
    fi
else
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/src.trn > ${dir}/src.wrd.trn
    if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref1.trn >> ${dir}/ref1.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref2.trn >> ${dir}/ref2.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref3.trn >> ${dir}/ref3.wrd.trn
    fi
fi

# detokenize
detokenizer.perl -l en -q < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
detokenizer.perl -l en -q < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok
detokenizer.perl -l en -q < ${dir}/src.wrd.trn > ${dir}/src.wrd.trn.detok
if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
    detokenizer.perl -l en -q < ${dir}/ref1.wrd.trn > ${dir}/ref1.wrd.trn.detok
    detokenizer.perl -l en -q < ${dir}/ref2.wrd.trn > ${dir}/ref2.wrd.trn.detok
    detokenizer.perl -l en -q < ${dir}/ref3.wrd.trn > ${dir}/ref3.wrd.trn.detok
fi

if [ -n "${nlsyms}" ]; then
    cp ${dir}/ref.wrd.trn.detok ${dir}/ref.wrd.trn.detok.org
    cp ${dir}/hyp.wrd.trn.detok ${dir}/hyp.wrd.trn.detok.org
    cp ${dir}/src.wrd.trn.detok ${dir}/src.wrd.trn.detok.org
    filt.py -v $nlsyms ${dir}/ref.wrd.trn.detok.org > ${dir}/ref.wrd.trn.detok
    filt.py -v $nlsyms ${dir}/hyp.wrd.trn.detok.org > ${dir}/hyp.wrd.trn.detok
    filt.py -v $nlsyms ${dir}/src.wrd.trn.detok.org > ${dir}/src.wrd.trn.detok
    if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
        cp ${dir}/ref1.wrd.trn.detok ${dir}/ref1.wrd.trn.detok.org
        cp ${dir}/ref2.wrd.trn.detok ${dir}/ref2.wrd.trn.detok.org
        cp ${dir}/ref3.wrd.trn.detok ${dir}/ref3.wrd.trn.detok.org
        filt.py -v $nlsyms ${dir}/ref1.wrd.trn.detok.org > ${dir}/ref1.wrd.trn.detok
        filt.py -v $nlsyms ${dir}/ref2.wrd.trn.detok.org > ${dir}/ref2.wrd.trn.detok
        filt.py -v $nlsyms ${dir}/ref3.wrd.trn.detok.org > ${dir}/ref3.wrd.trn.detok
    fi
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.wrd.trn.detok
    sed -i.bak3 -f ${filter} ${dir}/ref.wrd.trn.detok
    sed -i.bak3 -f ${filter} ${dir}/src.wrd.trn.detok
    if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
        sed -i.bak3 -f ${filter} ${dir}/ref1.wrd.trn.detok
        sed -i.bak3 -f ${filter} ${dir}/ref2.wrd.trn.detok
        sed -i.bak3 -f ${filter} ${dir}/ref3.wrd.trn.detok
    fi
fi

if [ ${case} = tc ]; then
    echo ${set} > ${dir}/result.tc.txt
    if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
        # 4 references
        sacrebleu ${dir}/ref.wrd.trn.detok ${dir}/ref1.wrd.trn.detok ${dir}/ref2.wrd.trn.detok ${dir}/ref3.wrd.trn.detok \
            -i ${dir}/hyp.wrd.trn.detok -m bleu chrf ter \
        >> ${dir}/result.tc.txt
    else
        # 1 reference
        sacrebleu ${dir}/ref.wrd.trn.detok -i ${dir}/hyp.wrd.trn.detok -m bleu chrf ter >> ${dir}/result.tc.txt
    fi
    echo "write a case-sensitive BLEU result in ${dir}/result.tc.txt"
    cat ${dir}/result.tc.txt
fi

# detokenize & remove punctuation except apostrophe
remove_punctuation.pl < ${dir}/ref.wrd.trn.detok > ${dir}/ref.wrd.trn.detok.lc.rm
remove_punctuation.pl < ${dir}/hyp.wrd.trn.detok > ${dir}/hyp.wrd.trn.detok.lc.rm
remove_punctuation.pl < ${dir}/src.wrd.trn.detok > ${dir}/src.wrd.trn.detok.lc.rm
if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
    remove_punctuation.pl < ${dir}/ref1.wrd.trn.detok > ${dir}/ref1.wrd.trn.detok.lc.rm
    remove_punctuation.pl < ${dir}/ref2.wrd.trn.detok > ${dir}/ref2.wrd.trn.detok.lc.rm
    remove_punctuation.pl < ${dir}/ref3.wrd.trn.detok > ${dir}/ref3.wrd.trn.detok.lc.rm
fi

echo ${set} > ${dir}/result.lc.txt
if [ -n "${set}" ] && [ -f ${dir}/data_ref1.json ]; then
    # 4 references
    echo "4-ref BLEU" >> ${dir}/result.lc.txt
    echo "########################################################################################################################" >> ${dir}/result.lc.txt
    echo "sacleBLEU" >> ${dir}/result.lc.txt
    sacrebleu -lc ${dir}/ref.wrd.trn.detok.lc.rm ${dir}/ref1.wrd.trn.detok.lc.rm ${dir}/ref2.wrd.trn.detok.lc.rm ${dir}/ref3.wrd.trn.detok.lc.rm \
        -i ${dir}/hyp.wrd.trn.detok.lc.rm -m bleu chrf ter \
        >> ${dir}/result.lc.txt
    echo "########################################################################################################################" >> ${dir}/result.lc.txt
fi
# 1 reference
echo "1-ref BLEU" >> ${dir}/result.lc.txt
echo "########################################################################################################################" >> ${dir}/result.lc.txt
echo "sacleBLEU" >> ${dir}/result.lc.txt
sacrebleu -lc ${dir}/ref.wrd.trn.detok.lc.rm -i ${dir}/hyp.wrd.trn.detok.lc.rm -m bleu chrf ter >> ${dir}/result.lc.txt
echo "########################################################################################################################" >> ${dir}/result.lc.txt
echo "write a case-insensitive BLEU result in ${dir}/result.lc.txt"
cat ${dir}/result.lc.txt

# TODO(hirofumi): add METEOR, BERTscore here
