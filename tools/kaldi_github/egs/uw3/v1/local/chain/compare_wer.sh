#!/bin/bash

# this script is used for comparing decoding results between systems.
# e.g. local/chain/compare_wer.sh exp/chain/cnn{1a,1b}

# ./local/chain/compare_wer.sh exp/chain/cnn1a
# System                          cnn1a
# WER                              0.61
# CER                              0.15
# Final train prob              -0.0377
# Final valid prob              -0.0380
# Final train prob (xent)       -0.0830
# Final valid prob (xent)       -0.0838

if [ $# == 0 ]; then
  echo "Usage: $0: <dir1> [<dir2> ... ]"
  echo "e.g.: $0 exp/chain/cnn{1a,1b}"
  exit 1
fi

echo "# $0 $*"
used_epochs=false

echo -n "# System                     "
for x in $*; do   printf "% 10s" " $(basename $x)";   done
echo

echo -n "# WER                        "
for x in $*; do
  wer=$(cat $x/decode_test/scoring_kaldi/best_wer | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# CER                        "
for x in $*; do
  cer=$(cat $x/decode_test/scoring_kaldi/best_cer | awk '{print $2}')
  printf "% 10s" $cer
done
echo

if $used_epochs; then
  exit 0;  # the diagnostics aren't comparable between regular and discriminatively trained systems.
fi

echo -n "# Final train prob           "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob           "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final train prob (xent)    "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob (xent)    "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo
