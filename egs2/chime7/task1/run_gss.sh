#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=4 # adjust based on number of your GPUs
affix=""
stage=1
stop_stage=100

. ./path.sh
. parse_options.sh

# Append _ to affix if not empty
affix=${affix:+_$affix}

MANIFEST_DIR=
DSET_NAME=CHiME6
DSET_PART=dev
EXP_DIR=
cmd=run.pl #if you use gridengine: "queue-freegpu.pl --gpu 1 --mem 8G --config conf/gpu.conf"
MAX_BATCH_DUR=90 # adjust based on your GPU VRAM

mkdir -p ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare cut set"
  lhotse cut simple --force-eager \
      -r ${MANIFEST_DIR}/${DSET_NAME}/${DSET_NAME}-mdm_recordings_${DSET_PART}.jsonl.gz \
      -s ${MANIFEST_DIR}/${DSET_NAME}/${DSET_NAME}-mdm_supervisions_${DSET_PART}.jsonl.gz \
      ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/cuts.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Trim cuts to supervisions (1 cut per supervision segment)"
  lhotse cut trim-to-supervisions --discard-overlapping \
       ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/cuts.jsonl.gz  \
       ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/cuts_per_segment.jsonl.gz
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Split segments into $nj parts"
  for part in dev eval; do
    gss utils split $nj  ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/cuts_per_segment.jsonl.gz \
     ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/split$nj
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Enhance segments using GSS (outer array mics)"
  $cmd JOB=1:$nj  ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/log/enhance.JOB.log \
    gss enhance cuts \
      ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/cuts.jsonl.gz  ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/split$nj/cuts_per_segment.JOB.jsonl.gz \
       ${EXP_DIR}/${DSET_NAME}_gss/${DSET_PART}/enhanced \
      --bss-iterations 20 \
      --context-duration 15.0 \
      --use-garbage-class \
      --min-segment-length 0.0 \
      --max-segment-length 20.0 \
      --max-batch-duration $MAX_BATCH_DUR \
      --max-batch-cuts 1 \
      --num-buckets 4 \
      --num-workers 4 \
      --force-overwrite \
      --duration-tolerance 3.0 || exit 1
  done
fi
