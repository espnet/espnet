#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=4 # adjust based on number of your GPUs
stage=1
stop_stage=100

manifests_dir=
dset_name=
dset_part=
exp_dir=
cmd=run.pl #if you use gridengine: "queue-freegpu.pl --gpu 1 --mem 8G --config conf/gpu.conf"
max_batch_dur= # adjust based on your GPU VRAM, here 40GB

. ./path.sh
. parse_options.sh

mkdir -p ${exp_dir}/${dset_name}/${dset_part}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare cut set"
  lhotse cut simple --force-eager \
      -r ${manifests_dir}/${dset_name}/${dset_part}/${dset_name}-mdm_recordings_${dset_part}.jsonl.gz \
      -s ${manifests_dir}/${dset_name}/${dset_part}/${dset_name}-mdm_supervisions_${dset_part}.jsonl.gz \
      ${exp_dir}/${dset_name}/${dset_part}/cuts.jsonl.gz
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Trim cuts to supervisions (1 cut per supervision segment)"
  lhotse cut trim-to-supervisions --discard-overlapping \
       ${exp_dir}/${dset_name}/${dset_part}/cuts.jsonl.gz  \
       ${exp_dir}/${dset_name}/${dset_part}/cuts_per_segment.jsonl.gz
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Split segments into $nj parts"
  for part in dev eval; do
    gss utils split $nj  ${exp_dir}/${dset_name}/${dset_part}/cuts_per_segment.jsonl.gz \
     ${exp_dir}/${dset_name}/${dset_part}/split$nj
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Enhance segments using GSS"
  $cmd JOB=1:$nj  ${exp_dir}/${dset_name}/${dset_part}/log/enhance.JOB.log \
    gss enhance cuts \
      ${exp_dir}/${dset_name}/${dset_part}/cuts.jsonl.gz  ${exp_dir}/${dset_name}/${dset_part}/split$nj/cuts_per_segment.JOB.jsonl.gz \
       ${exp_dir}/${dset_name}/${dset_part}/enhanced \
      --bss-iterations 20 \
      --context-duration 15.0 \
      --use-garbage-class \
      --min-segment-length 0.0 \
      --max-segment-length 20.0 \
      --max-batch-duration $max_batch_dur \
      --max-batch-cuts 1 \
      --num-buckets 4 \
      --num-workers 4 \
      --force-overwrite \
      --duration-tolerance 3.0 || exit 1
fi
