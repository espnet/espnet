#!/usr/bin/env bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Copyright 2019       Johns Hopkins University (Author: Shinji Watanabe)
# Copyright 2020       University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0
#
# This script scores the multi-speaker LibriCSS recordings.

cmd=run.pl

conditions="0L 0S OV10 OV20 OV30 OV40"

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
    echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <decode-dir>"
    echo "This script scores the LibriCSS full recordings"
    echo " Options:"
    echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
    exit 1;
fi

dir=$1

ark="perl -p -e 's/(.*?)\\(([^)]+)\\)\$/\\\$2 \\\$1/g'"

echo "Evaluating $dir"

# get the scoring result per utterance. Copied from local/score.sh
$cmd $dir/log/stats_eval.log \
     align-text --special-symbol="'***'" "ark:$ark $dir/ref.wrd.trn |" "ark:$ark $dir/hyp.wrd.trn |" ark,t:- \|  \
     utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \> $dir/per_utt

score_result=$dir/per_utt

for cond in $conditions; do
    # get nerror
    nerr=`grep "\#csid" $score_result | grep $cond | awk '{sum+=$4+$5+$6} END {print sum}'`
    # get nwords from references (NF-2 means to exclude utterance id and " ref ")
    nwrd=`grep "\#csid" $score_result | grep $cond | awk '{sum+=$3+$4+$6} END {print sum}'`
    # compute wer with scale=2
    wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`

    # report the results
    echo -n "Condition $cond: "
    echo -n "#words $nwrd, "
    echo -n "#errors $nerr, "
    echo "wer $wer %"
done

echo -n "overall: "
# get nerror
nerr=`grep "\#csid" $score_result | awk '{sum+=$4+$5+$6} END {print sum}'`
# get nwords from references (NF-2 means to exclude utterance id and " ref ")
nwrd=`grep "\#csid" $score_result | awk '{sum+=$3+$4+$6} END {print sum}'`
# compute wer with scale=2
wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`
echo -n "overall: "
echo -n "#words $nwrd, "
echo -n "#errors $nerr, "
echo "wer $wer %"
