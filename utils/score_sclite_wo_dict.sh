#!/usr/bin/env bash

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
    json2trn_wo_dict.py ${dir}/data.json --num-spkrs ${num_spkrs} --refs ${dir}/ref_org.wrd.trn --hyps ${dir}/hyp_org.wrd.trn

    cat < ${dir}/hyp_org.wrd.trn | sed -e 's/▁//' | sed -e 's/▁/ /g' > ${dir}/hyp.wrd.trn
    cat < ${dir}/ref_org.wrd.trn | sed -e 's/\.//g' -e 's/\,//g' > ${dir}/ref.wrd.trn

    cat < ${dir}/hyp.wrd.trn | awk -v FS='' '{a=0;for(i=1;i<=NF;i++){if($i=="("){a=1};if(a==0){printf("%s ",$i)}else{printf("%s",$i)}}printf("\n")}' > ${dir}/hyp.trn
    cat < ${dir}/ref.wrd.trn | awk -v FS='' '{a=0;for(i=1;i<=NF;i++){if($i=="("){a=1};if(a==0){printf("%s ",$i)}else{printf("%s",$i)}}printf("\n")}' > ${dir}/ref.trn

    sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn -i rm -o all stdout > ${dir}/result.txt
    echo "write a CER result in ${dir}/result.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.txt

    if ${wer}; then
        sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn -i rm -o all stdout > ${dir}/result.wrd.txt
        echo "write a WER result in ${dir}/result.wrd.txt"
        grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt

        sclite -r ${dir}/ref_org.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result_w_punc.wrd.txt
        echo "write a WER result in ${dir}/result_w_punc.wrd.txt"
        grep -e Avg -e SPKR -m 2 ${dir}/result_w_punc.wrd.txt

    fi
fi
