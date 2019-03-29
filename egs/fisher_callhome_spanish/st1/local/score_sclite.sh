#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
case=lc.rm

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn.py ${dir}/data.json ${dic} --refs ${dir}/ref.trn --hyps ${dir}/hyp.trn

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

# case-sensitive
if [ ${case} = tc ]; then
  sclite -s -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.tc.txt

  echo "write a case-sensitive CER (or TER) result in ${dir}/result.tc.txt"
  grep -e Avg -e SPKR -m 2 ${dir}/result.tc.txt

  if ${wer}; then
      if [ -n "$bpe" ]; then
          spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
          spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
      else
          sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
          sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
      fi
      sclite -s -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.tc.txt

      echo "write a case-sensitive WER result in ${dir}/result.wrd.tc.txt"
      grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.tc.txt
  fi
fi

# remove punctuation
paste -d "(" <(cut -d '(' -f 1 ${dir}/hyp.trn | local/remove_punctuation.pl | sed -e "s/¿//g" | sed -e "s/  / /g") <(cut -d '(' -f 2- ${dir}/hyp.trn) \
    > ${dir}/hyp.trn.lc.rm
paste -d "(" <(cut -d '(' -f 1 ${dir}/ref.trn | local/remove_punctuation.pl | sed -e "s/¿//g" | sed -e "s/  / /g") <(cut -d '(' -f 2- ${dir}/ref.trn) \
    > ${dir}/ref.trn.lc.rm

sclite -r ${dir}/ref.trn.lc.rm trn -h ${dir}/hyp.trn.lc.rm trn -i rm -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt

if ${wer}; then
    if [ -n "$bpe" ]; then
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn.lc.rm | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn.lc.rm
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn.lc.rm | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn.lc.rm
    else
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn.lc.rm > ${dir}/ref.wrd.trn.lc.rm
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn.lc.rm > ${dir}/hyp.wrd.trn.lc.rm
    fi
    sclite -r ${dir}/ref.wrd.trn.lc.rm trn -h ${dir}/hyp.wrd.trn.lc.rm trn -i rm -o all stdout > ${dir}/result.wrd.txt

    echo "write a WER result in ${dir}/result.wrd.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
