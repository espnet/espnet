#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=4
affix=""
stage=0
stop_stage=100

. ./path.sh
. parse_options.sh

# Append _ to affix if not empty
affix=${affix:+_$affix}

CORPUS_DIR=/raid/users/popcornell/CHiME6/espnet/egs2/chime6/asr1/CHiME6/
DATA_DIR=data/
EXP_DIR=exp/chime6${affix}

cmd="run.pl --gpu 1 --mem 8G" #"queue-freegpu.pl --gpu 1 --mem 8G --config conf/gpu.conf"

mkdir -p $DATA_DIR
mkdir -p $EXP_DIR/{dev,eval}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Stage 0: Prepare manifests"
  lhotse prepare chime6 --mic mdm -p dev -p eval $CORPUS_DIR $DATA_DIR
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare cut set"
  for part in dev eval; do
    lhotse cut simple --force-eager \
      -r $DATA_DIR/chime6-mdm_recordings_${part}.jsonl.gz \
      -s $DATA_DIR/chime6-mdm_supervisions_${part}.jsonl.gz \
      $EXP_DIR/$part/cuts.jsonl.gz
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Trim cuts to supervisions (1 cut per supervision segment)"
  for part in dev eval; do
    lhotse cut trim-to-supervisions --discard-overlapping \
      $EXP_DIR/$part/cuts.jsonl.gz $EXP_DIR/$part/cuts_per_segment.jsonl.gz
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Split segments into $nj parts"
  for part in dev eval; do
    gss utils split $nj $EXP_DIR/$part/cuts_per_segment.jsonl.gz $EXP_DIR/$part/split$nj
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Enhance segments using GSS (outer array mics)"
  # NOTE: U03 is missing is S01 and U05 is missing in S09, so we only use
  # 10 channels here instead of 12.
  for part in dev; do
    $cmd JOB=1:$nj $EXP_DIR/$part/log/enhance.JOB.log \
      gss enhance cuts \
        $EXP_DIR/$part/cuts.jsonl.gz $EXP_DIR/$part/split$nj/cuts_per_segment.JOB.jsonl.gz \
        $EXP_DIR/$part/enhanced \
        --bss-iterations 20 \
        --context-duration 15.0 \
        --use-garbage-class \
        --min-segment-length 0.0 \
        --max-segment-length 20.0 \
        --max-batch-duration 90.0 \
        --max-batch-cuts 1 \
        --num-buckets 4 \
        --num-workers 4 \
        --force-overwrite \
        --duration-tolerance 3.0 || exit 1
  done
fi
