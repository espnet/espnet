#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
filter=""
remove_nonverbal=true

src_lang=en
tgt_lang=de

# submission related
system="primary"
user_id="espnet-st-group"

. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: $0 <decode-dir> <dict> <data-dir> <set>";
    exit 1;
fi

dir=$1
dict=$2
set=$4
src=$3/${set}/IWSLT.${set}

# sort hypotheses
concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn_reorder.py ${dir}/data.json ${dict} ${dir}/hyp.trn.org ${src}/FILE_ORDER

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn

# remove non-verbal labels (optional)
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn > ${dir}/hyp.rm.trn

if [ -n "$bpe" ]; then
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
detokenizer.perl -l ${tgt_lang} -q < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok

# remove language IDs
if [ -n "${nlsyms}" ]; then
    cp ${dir}/hyp.wrd.trn.detok ${dir}/hyp.wrd.trn.detok.tmp
    filt.py -v $nlsyms ${dir}/hyp.wrd.trn.detok.tmp > ${dir}/hyp.wrd.trn.detok
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.wrd.trn.detok
fi
# NOTE: this must be performed after detokenization so that punctuation marks are not removed


# pack output
task="${src_lang}-${tgt_lang}"
file_name=IWSLT21.SLT.${set}.${task}.${user_id}.${system}.txt
submission_dir=${user_id}
mkdir -p ${submission_dir}
cp ${dir}/hyp.wrd.trn.detok ${submission_dir}/${file_name}
