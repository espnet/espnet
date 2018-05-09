#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2017  Johns Hopkins University (Author: Shinji Watanabe)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
stft_config=conf/stft.conf
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <data-dir> <log-dir> <path-to-stftdir>";
   echo "e.g.: $0 data/train exp/make_stft/train stft"
   echo "options: "
   echo "options: "
   echo "  --stft-config <config-file>                      # config passed to compute-stft-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
stftdir=$3


# make ${stftdir} an absolute pathname.
stftdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${stftdir} ${PWD}`

# use "name" as part of name of the archive.
name=`basename ${data}`

mkdir -p ${stftdir} || exit 1;
mkdir -p ${logdir} || exit 1;

if [ -f ${data}/feats.scp ]; then
  mkdir -p ${data}/.backup
  echo "$0: moving ${data}/feats.scp to ${data}/.backup"
  mv ${data}/feats.scp ${data}/.backup
fi

scp=${data}/wav.scp

required="${scp} ${stft_config}"

for f in ${required}; do
  if [ ! -f ${f} ]; then
    echo "make_stft.sh: no such file ${f}"
    exit 1;
  fi
done
utils/validate_data_dir.sh --no-text --no-feats ${data} || exit 1;

for n in $(seq ${nj}); do
  # the next command does nothing unless ${stftdir}/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl ${stftdir}/raw_stft_${name}.${n}.ark
done


if [ -f ${data}/segments ]; then
  echo "$0 [info]: segments file exists: using that."

  split_segments=""
  for n in $(seq ${nj}); do
    split_segments="${split_segments} ${logdir}/segments.${n}"
  done

  utils/split_scp.pl ${data}/segments ${split_segments} || exit 1;
  rm ${logdir}/.error 2>/dev/null

  # compute-stft-feats.py does not support kaldi reader, and
  # we have to dump segment files
  ${cmd} JOB=1:${nj} ${logdir}/make_segment_${name}.JOB.log \
    extract-segments scp,p:${scp} ${logdir}/segments.JOB \
      ark,scp:${stftdir}/raw_segment_${name}.JOB.ark,${stftdir}/raw_segment_${name}.JOB.scp \
     || exit 1;
  ${cmd} JOB=1:${nj} ${logdir}/make_stft_${name}.JOB.log \
    compute-stft-feats.py --config=${stft_config} ${stftdir}/raw_segment_${name}.JOB.scp ark:- \| \
    copy-feats --compress=${compress} ark:- \
      ark,scp:${stftdir}/raw_stft_${name}.JOB.ark,${stftdir}/raw_stft_${name}.JOB.scp \
     || exit 1;
  rm -f ${stftdir}/raw_segment_${name}.*
else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav_${name}.${n}.scp"
  done

  utils/split_scp.pl ${scp} ${split_scps} || exit 1;


  # add ,p to the input rspecifier so that we can just skip over
  # utterances that have bad wave data.

  ${cmd} JOB=1:${nj} ${logdir}/make_stft_${name}.JOB.log \
    compute-stft-feats.py --config=${stft_config} \
      ${logdir}/wav_${name}.JOB.scp ark:- \| \
      copy-feats --compress=${compress} ark:- \
      ark,scp:${stftdir}/raw_stft_${name}.JOB.ark,${stftdir}/raw_stft_${name}.JOB.scp \
      || exit 1;
fi


if [ -f ${logdir}/.error.${name} ]; then
  echo "Error producing stft features for ${name}:"
  tail ${logdir}/make_stft_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq ${nj}); do
  cat ${stftdir}/raw_stft_${name}.${n}.scp || exit 1;
done > ${data}/feats.scp

rm ${logdir}/wav_${name}.*.scp  ${logdir}/segments.* 2>/dev/null

nf=`cat ${data}/feats.scp | wc -l`
nu=`cat ${data}/utt2spk | wc -l`
if [ ${nf} -ne ${nu} ]; then
  echo "It seems not all of the feature files were successfully processed (${nf} != ${nu});"
  echo "consider using utils/fix_data_dir.sh ${data}"
fi

if [ ${nf} -lt $[${nu} - (${nu}/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating stft features for ${name}"
