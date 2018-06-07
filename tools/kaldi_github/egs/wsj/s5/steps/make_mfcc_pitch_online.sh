#!/bin/bash

# Copyright 2013 The Shenzhen Key Laboratory of Intelligent Media and Speech,
#                PKU-HKUST Shenzhen Hong Kong Institution (Author: Wei Shi)
#           2014-2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# Combine MFCC and online-pitch features together
# Note: This file is based on make_mfcc_pitch.sh

# Begin configuration section.
nj=4
cmd=run.pl
mfcc_config=conf/mfcc.conf
online_pitch_config=conf/online_pitch.conf
paste_length_tolerance=2
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<mfcc-dir>] ]";
   echo "e.g.: $0 data/train exp/make_mfcc/train mfcc"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <mfcc-dir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --mfcc-config              <mfcc-config-file>        # config passed to compute-mfcc-feats, default "
   echo "                                                       # is conf/mfcc.conf"
   echo "  --online-pitch-config <online-pitch-config-file>     # config passed to compute-and-process-kaldi-pitch-feats, "
   echo "                                                       # default is conf/online_pitch.conf"
   echo "  --paste-length-tolerance   <tolerance>               # length tolerance passed to paste-feats"
   echo "  --nj                       <nj>                      # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>)     # how to run jobs."
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  mfcc_pitch_dir=$3
else
  mfcc_pitch_dir=$data/data
fi


# make $mfcc_pitch_dir an absolute pathname.
mfcc_pitch_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfcc_pitch_dir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $mfcc_pitch_dir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $mfcc_config $online_pitch_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_mfcc_pitch.sh: no such file $f"
    exit 1;
  fi
done
utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
elif [ -f $data/utt2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/utt2warp"
  vtln_opts="--vtln-map=ark:$data/utt2warp"
fi

for n in $(seq $nj); do
  # the next command does nothing unless $mfcc_pitch_dir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $mfcc_pitch_dir/raw_mfcc_online_pitch_$name.$n.ark
done

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  mfcc_feats="ark:extract-segments scp,p:$scp $logdir/segments.JOB ark:- | compute-mfcc-feats $vtln_opts --verbose=2 --config=$mfcc_config ark:- ark:- |"
  pitch_feats="ark,s,cs:extract-segments scp,p:$scp $logdir/segments.JOB ark:- | compute-and-process-kaldi-pitch-feats --verbose=2 --config=$online_pitch_config ark:- ark:- |"

  $cmd JOB=1:$nj $logdir/make_mfcc_pitch_${name}.JOB.log \
    paste-feats --length-tolerance=$paste_length_tolerance "$mfcc_feats" "$pitch_feats" ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$mfcc_pitch_dir/raw_mfcc_online_pitch_$name.JOB.ark,$mfcc_pitch_dir/raw_mfcc_online_pitch_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;

  mfcc_feats="ark:compute-mfcc-feats $vtln_opts --verbose=2 --config=$mfcc_config scp,p:$logdir/wav_${name}.JOB.scp ark:- |"
  pitch_feats="ark,s,cs:compute-and-process-kaldi-pitch-feats --verbose=2 --config=$online_pitch_config scp,p:$logdir/wav_${name}.JOB.scp ark:- |"

  $cmd JOB=1:$nj $logdir/make_mfcc_pitch_${name}.JOB.log \
    paste-feats --length-tolerance=$paste_length_tolerance "$mfcc_feats" "$pitch_feats" ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$mfcc_pitch_dir/raw_mfcc_online_pitch_$name.JOB.ark,$mfcc_pitch_dir/raw_mfcc_online_pitch_$name.JOB.scp \
      || exit 1;
fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing mfcc & pitch features for $name:"
  tail $logdir/make_mfcc_pitch_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $mfcc_pitch_dir/raw_mfcc_online_pitch_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav_${name}.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating MFCC & online-pitch features for $name"
