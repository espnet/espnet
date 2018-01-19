#!/bin/bash

# Copyright 2017 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

cmd=run.pl
do_delta=false

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

# dump features 
if ${do_delta};then
    $cmd $logdir/dump_feature.log \
        apply-cmvn --norm-vars=true $cvmnark scp:$scp ark:- \| \
        add-deltas ark:- ark:- \| \
        copy-feats ark:- ark,t:- \| \
        local/make_hdf5.py -w $dumpdir/feats.h5
else
    $cmd $logdir/dump_feature.log \
        apply-cmvn --norm-vars=true $cvmnark scp:$scp ark:- \| \
        copy-feats ark:- ark,t:- \| \
        local/make_hdf5.py -w $dumpdir/feats.h5
fi

echo "Succeeded dumping features for training"
