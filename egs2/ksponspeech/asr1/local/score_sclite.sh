#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2020 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
space_norm=false
filter=""
num_spkrs=1
help_message="Usage: $0 <data-dir> <dict>"

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json

if [ $num_spkrs -eq 1 ]; then
  json2trn.py ${dir}/data.json ${dic} --num-spkrs ${num_spkrs} --refs ${dir}/ref.trn --hyps ${dir}/hyp.trn

  if ${remove_blank}; then
      sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
  fi
  if [ -n "${nlsyms}" ]; then
      cp ${dir}/ref.trn ${dir}/ref.trn.org
      cp ${dir}/hyp.trn ${dir}/hyp.trn.org
      filt.py -v ${nlsyms} ${dir}/ref.trn.org > ${dir}/ref.trn
      filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
  fi
  if [ -n "${filter}" ]; then
      sed -i.bak3 -f ${filter} ${dir}/hyp.trn
      sed -i.bak3 -f ${filter} ${dir}/ref.trn
  fi

  sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.txt

  echo "write a CER (or TER) result in ${dir}/result.txt"
  grep -e Avg -e SPKR -m 2 ${dir}/result.txt

  if ${wer}; then
      if [ -n "$bpe" ]; then
  	    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
  	    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
      else
  	    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
  	    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
      fi

      sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt
      echo "write a WER result in ${dir}/result.wrd.txt"
      grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt

      if $space_norm; then
            # get space-normalized texts
            python local/get_space_normalized_hyps.py --verbose 0 \
                  --in-ref ${dir}/ref.wrd.trn --in-hyp ${dir}/hyp.wrd.trn \
                  --out-ref ${dir}/ref.sp_norm.trn --out-hyp ${dir}/hyp.sp_norm.trn || exit 1;

            # character error rate; CER (excluding space symbols)
            sclite -r ${dir}/ref.sp_norm.trn trn -h ${dir}/hyp.sp_norm.trn trn -i rm -o all stdout > ${dir}/result.sp_norm.txt
            echo "write a CER result in ${dir}/result.sp_norm.txt"
            grep -e Avg -e SPKR -m 2 ${dir}/result.sp_norm.txt

            # space-normalized word error rate; sWER
            cat ${dir}/ref.sp_norm.trn | sed -e "s/ //g" | sed -e "s/▁/ /g" > ${dir}/ref.wrd.sp_norm.trn
            cat ${dir}/hyp.sp_norm.trn | sed -e "s/ //g" | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.sp_norm.trn
            sclite -r ${dir}/ref.wrd.sp_norm.trn trn -h ${dir}/hyp.wrd.sp_norm.trn trn -i rm -o all stdout > ${dir}/result.wrd.sp_norm.txt
            echo "write a sWER result in ${dir}/result.wrd.sp_norm.txt"
            grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.sp_norm.txt
      fi
  fi
fi
