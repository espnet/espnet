#!/bin/bash

# Copyright 2017 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

cmd=run.pl
do_delta=false
nj=1
verbose=0
compress=true
write_utt2num_frames=true
ivectors=""

. utils/parse_options.sh

scp=$1
cvmnark=$2
logdir=$3
dumpdir=$4

if [ $# != 4 ]; then
    echo "Usage: $0 <scp> <cmvnark> <logdir> <dumpdir>"
    exit 1;
fi

mkdir -p $logdir
mkdir -p $dumpdir

dumpdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${dumpdir} ${PWD}`

for n in $(seq $nj); do
    # the next command does nothing unless $dumpdir/storage/ exists, see
    # utils/create_data_link.pl for more info.
    utils/create_data_link.pl ${dumpdir}/feats.${n}.ark
done

if $write_utt2num_frames; then
    write_num_frames_opt="--write-num-frames=ark,t:$dumpdir/utt2num_frames.JOB"
else
    write_num_frames_opt=
fi

# split scp file
split_scps=""
split_scps_ivectors=""
for n in $(seq $nj); do
    split_scps="$split_scps $logdir/feats.$n.scp"
    if [ ! -z $ivectors ]; then
        split_scps_ivectors="$split_scps_ivectors $logdir/online_ivectors.$n.scp"
    fi
done

utils/split_scp.pl $scp $split_scps || exit 1;
if [ ! -z $ivectors ]; then
    utils/split_scp.pl $ivectors $split_scps_ivectors || exit 1;
fi

# dump features
if  [ ! -z ${ivectors} ]; then
    $cmd JOB=1:$nj $logdir/dump_feature.JOB.log \
        paste-feats --length-tolerance=10 scp:$logdir/feats.JOB.scp "ark:subsample-feats --n=-10 scp:$logdir/online_ivectors.JOB.scp ark:- |" ark:- \| \
        copy-feats --compress=$compress --compression-method=2 ${write_num_frames_opt} \
            ark:- ark,scp:${dumpdir}/feats.JOB.ark,${dumpdir}/feats.JOB.scp \
        || exit 1
elif ${do_delta};then
    $cmd JOB=1:$nj $logdir/dump_feature.JOB.log \
        apply-cmvn --norm-vars=true $cvmnark scp:$logdir/feats.JOB.scp ark:- \| \
        add-deltas ark:- ark:- \| \
        copy-feats --compress=$compress --compression-method=2 ${write_num_frames_opt} \
            ark:- ark,scp:${dumpdir}/feats.JOB.ark,${dumpdir}/feats.JOB.scp \
        || exit 1
else
    $cmd JOB=1:$nj $logdir/dump_feature.JOB.log \
        paste-feats scp:$logdir/feats.JOB.scp scp:$logdir/ark:- \| \

        apply-cmvn --norm-vars=true $cvmnark scp:$logdir/feats.JOB.scp ark:- \| \
        copy-feats --compress=$compress --compression-method=2 ${write_num_frames_opt} \
            ark:- ark,scp:${dumpdir}/feats.JOB.ark,${dumpdir}/feats.JOB.scp \
        || exit 1
fi

# concatenate scp files
for n in $(seq $nj); do
    cat $dumpdir/feats.$n.scp || exit 1;
done > $dumpdir/feats.scp || exit 1

if $write_utt2num_frames; then
    for n in $(seq $nj); do
        cat $dumpdir/utt2num_frames.$n || exit 1;
    done > $dumpdir/utt2num_frames || exit 1
    rm $dumpdir/utt2num_frames.* 2>/dev/null
fi

# remove temp scps
rm $logdir/feats.*.scp 2>/dev/null
if [ ${verbose} -eq 1 ]; then
    echo "Succeeded dumping features for training"
fi
