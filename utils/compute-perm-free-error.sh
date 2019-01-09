#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

num_spkrs=2
wrd=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
    echo "Usage: $0 [options] <result_r1h1> <result_r1h2> <result_r2h1> <result_r2h2>";
    echo "Options: "
    echo "  --num_spkrs <num_spkrs>         # number of speakers (Now, only 2 speakers are supported)"
    echo "  --wrd <wrd>                     # word results or not"
    exit 1;
fi

r1h1=$1
r1h2=$2
r2h1=$3
r2h2=$4
dir=`dirname $r1h1`
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
rm -rf ${tmpdir}/*.json

# convert all the results to json
for f in $r1h1 $r1h2 $r2h1 $r2h2; do
    k=`basename $f .txt`
    cat $f | result2json.py --key $k > ${tmpdir}/${k}.json
done

mergeresultjson.py ${tmpdir}/*.json > ${tmpdir}/all_result${wrd:+.wrd}.json
# compute the WER and select the best permutation
minpermwer.py ${tmpdir}/all_result${wrd:+.wrd}.json

rm -r $tmpdir
