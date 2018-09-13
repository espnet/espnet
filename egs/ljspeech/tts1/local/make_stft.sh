#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Begin configuration section.
nj=4
fs=22050
n_fft=1024
n_shift=512
win_length=
write_utt2num_frames=true
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]";
   echo "e.g.: $0 data/train exp/make_stft/train stft"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  fbankdir=$3
else
  fbankdir=$data/data
fi

# make $fbankdir an absolute pathname.
fbankdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $fbankdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $fbankdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

split_scps=""
for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav.$n.scp"
done

utils/split_scp.pl $scp $split_scps || exit 1;

$cmd JOB=1:$nj $logdir/make_stft_${name}.JOB.log \
    local/compute-stft-feats.py \
        --fs $fs \
        --win_length $win_length \
        --n_fft $n_fft \
        --n_shift $n_shift \
        --write_utt2num_frames ${write_utt2num_frames} \
        $logdir/wav.JOB.scp \
        $fbankdir/raw_stft_$name.JOB

# concatenate the .scp files together.
for n in $(seq $nj); do
    cat $fbankdir/raw_stft_$name.$n.scp || exit 1;
done > $data/feats.scp || exit 1

if $write_utt2num_frames; then
    for n in $(seq $nj); do
        cat $fbankdir/utt2num_frames.$n || exit 1;
    done > $data/utt2num_frames || exit 1
fi

rm $logdir/wav.*.scp 2>/dev/null

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/wav.scp | wc -l`
if [ $nf -ne $nu ]; then
    echo "It seems not all of the feature files were successfully ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
