#!/usr/bin/env bash
# Apache 2.0
#
# This script performs CHiME-6 track 2 style scoring for the diarized data.
# This means that all permutations of reference and hypothesis speakers are
# scored and the best one is selected to compute a kind of "speaker-attributed" WER.

cmd=run.pl

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

conditions="0L 0S OV10 OV20 OV30 OV40"

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <decode-dir> <data-dir>"
    echo "This script provides CHiME-6 style SA-WER scoring for LibriCSS"
    echo " Options:"
    echo "    --cmd (run.pl|queue.pl...)            # specify how to run the sub-processes."

    exit 1;
fi

decodedir=$1
datadir=$2

echo "Scoring $datadir"

# obtaining per recording stats
perl -p -e 's/(.*?)\s*\([^\-]+?-(.+)\)$/$2 $1/g' \
  $decodedir/hyp.wrd.trn | grep -vP '^\S+ $' > $decodedir/hyp.wrd.txt
local/multispeaker_score.sh --cmd "$cmd" \
  --datadir $datadir $datadir/text.bak \
  $decodedir/hyp.wrd.txt \
  $decodedir/scoring_kaldi_multispeaker/

find $decodedir/scoring_kaldi_multispeaker/per_speaker_wer -maxdepth 1 -name "wer_*" -delete

# Compute the average WER stats for all conditions individually.
wer_dir=$decodedir/scoring_kaldi_multispeaker/per_speaker_wer
for cond in $conditions; do
  grep $cond $wer_dir/best_wer_all | awk -v COND="$cond" '
    {
      ERR+=$5; WC+=$7; INS+=$8; DEL+=$10; SUB+=$12;
    }END{
      WER=ERR*100/WC;
      printf("%s %%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]\n",COND,WER,ERR,WC,INS,DEL,SUB);
    }
    ' >> $decodedir/scoring_kaldi_multispeaker/best_wer
done

# Compute overall WER average
cat $wer_dir/best_wer_all | awk '
  {
    ERR+=$5; WC+=$7; INS+=$8; DEL+=$10; SUB+=$12;
  }END{
    WER=ERR*100/WC;
    printf("%%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]",WER,ERR,WC,INS,DEL,SUB);
  }
  ' > $decodedir/scoring_kaldi_multispeaker/best_wer_average

# printing dev and eval wer
echo "$datadir WERs:"
cat $decodedir/scoring_kaldi_multispeaker/best_wer
echo "Average $datadir WER:"
cat $decodedir/scoring_kaldi_multispeaker/best_wer_average
echo
