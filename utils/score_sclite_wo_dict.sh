#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

wer=false
num_spkrs=1
help_message="Usage: $0 <data-dir>"

. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1

concatjson.py ${dir}/data.*.json > ${dir}/data.json

if [ $num_spkrs -eq 1 ]; then
    json2trn_wo_dict.py ${dir}/data.json --num-spkrs ${num_spkrs} --refs ${dir}/ref.trn --hyps ${dir}/hyp_org.trn
   
    cat ${dir}/hyp_org.trn | sed -e 's/▁//' | sed -e 's/▁/ /g' > ${dir}/hyp.trn
    cat ${dir}/ref.trn | sed -e 's/\.//g' -e 's/\,//g' > ${dir}/ref_wo_punc.trn

    if ${wer}; then
        sclite -r ${dir}/ref_wo_punc.trn trn -h ${dir}/hyp.trn -i rm -o all stdout > ${dir}/result_wo_punc.wrd.txt
        echo "write a WER result in ${dir}/result_wo_punc.wrd.txt"
        grep -e Avg -e SPKR -m 2 ${dir}/result_wo_punc.wrd.txt
        
        sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt
        echo "write a WER result in ${dir}/result.wrd.txt"
        grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt

    fi
fi


