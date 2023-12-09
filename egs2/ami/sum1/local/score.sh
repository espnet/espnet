#!/usr/bin/env bash
# Copyright 2021 Carnegie Mellon University (Author : Roshan Sharma)

## begin configuration section.
# end configuration section.


[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir>"
  exit 1;
fi


asr_expdir=$1

#name=$(basename ${data}) # e.g. dev5_test
#echo "${asr_expdir}/decode_*/${name}"
for dir in $( ls ${asr_expdir}/*/ | grep decode); do 
	dir=${dir::-1}
	for test_dir in $(ls ${dir}/*/ | grep eval); do
	       test_dir=${test_dir::-1}
	       echo "Dir $dir testdir ${test_dir} "
               name=$(basename ${test_dir})	       
    		python pyscripts/utils/score_summarization.py data/${name}/text ${test_dir}/text $(echo $dir | sed 's/exp//g') > ${test_dir}/result.sum
	done
done
