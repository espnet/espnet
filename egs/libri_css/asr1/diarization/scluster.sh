#!/usr/bin/env bash

# Copyright       2016  David Snyder
#            2017-2018  Matthew Maciejewski
#                 2020  Maxim Korenevsky (STC-innovations Ltd)
# Apache 2.0.

# This script performs spectral clustering using scored
# pairs of subsegments and produces a rttm file with speaker
# labels derived from the clusters.

# Begin configuration section.
cmd="run.pl"
stage=0
nj=10
cleanup=true
threshold=0.5
max_spk_fraction=1.0
first_pass_max_utterances=32767
rttm_channel=0
read_costs=false
reco2num_spk=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <src-dir> <dir>"
  echo " e.g.: $0 exp/ivectors_callhome exp/ivectors_callhome/results"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --threshold <threshold|0>                        # Cluster stopping criterion. Clusters with scores greater"
  echo "                                                   # than this value will be merged until all clusters"
  echo "                                                   # exceed this value."
  echo "  --max-spk-fraction <max-spk-fraction|1.0>        # Clusters with total fraction of utterances greater than"
  echo "                                                   # this value will not be merged. This is active only when"
  echo "                                                   # reco2num-spk is supplied and"
  echo "                                                   # 1.0 / num-spk <= max-spk-fraction <= 1.0."
  echo "  --first-pass-max-utterances <max-utts|32767>     # If the number of utterances is larger than first-pass-max-utterances,"
  echo "                                                   # then clustering is done in two passes. In the first pass, input points"
  echo "                                                   # are divided into contiguous subsets of size first-pass-max-utterances"
  echo "                                                   # and each subset is clustered separately. In the second pass, the first"
  echo "                                                   # pass clusters are merged into the final set of clusters."
  echo "  --rttm-channel <rttm-channel|0>                  # The value passed into the RTTM channel field. Only affects"
  echo "                                                   # the format of the RTTM file."
  echo "  --read-costs <read-costs|false>                  # If true, interpret input scores as costs, i.e. similarity"
  echo "                                                   # is indicated by smaller values. If enabled, clusters will"
  echo "                                                   # be merged until all cluster scores are less than the"
  echo "                                                   # threshold value."
  echo "  --reco2num-spk <reco2num-spk-file>               # File containing mapping of recording ID"
  echo "                                                   # to number of speakers. Used instead of threshold"
  echo "                                                   # as stopping criterion if supplied."
  echo "  --cleanup <bool|false>                           # If true, remove temporary files"
  exit 1;
fi

srcdir=$1
dir=$2

mkdir -p $dir/tmp

for f in $srcdir/scores.scp $srcdir/spk2utt $srcdir/utt2spk $srcdir/segments ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

cp $srcdir/spk2utt $dir/tmp/
cp $srcdir/utt2spk $dir/tmp/
cp $srcdir/segments $dir/tmp/
utils/fix_data_dir.sh $dir/tmp > /dev/null

if [ ! -z $reco2num_spk ]; then
  reco2num_spk="ark,t:$reco2num_spk"
fi

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

feats="utils/filter_scp.pl $sdata/JOB/spk2utt $srcdir/scores.scp |"

reco2num_spk_opt=
if [ ! $reco2num_spk == "" ]; then
  reco2num_spk_opt="--reco2num-spk $reco2num_spk"
fi

if [ $stage -le 0 ]; then
  echo "$0: clustering scores"
  for j in `seq $nj`; do 
    utils/filter_scp.pl $sdata/$j/spk2utt $srcdir/scores.scp > $dir/scores.$j.scp
  done
  $cmd JOB=1:$nj $dir/log/spectral_cluster.JOB.log \
    diarization/spec_clust.py $reco2num_spk_opt \
      scp:$dir/scores.JOB.scp ark,t:$sdata/JOB/spk2utt ark,t:$dir/labels.JOB || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining labels"
  for j in $(seq $nj); do cat $dir/labels.$j; done > $dir/labels || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  diarization/make_rttm.py --rttm-channel $rttm_channel $srcdir/segments $dir/labels $dir/rttm || exit 1;
fi

if $cleanup ; then
  rm -r $dir/tmp || exit 1;
fi
