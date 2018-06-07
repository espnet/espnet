#!/bin/bash

# Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.

# This script does decoding with a neural-net.  If the neural net was built on
# top of fMLLR transforms from a conventional system, you should provide the
# --transform-dir option.

# Begin configuration section.
stage=1
transform_dir=    # dir to find fMLLR transforms.
nj=4 # number of decoding jobs.  If --transform-dir set, must match that number!
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
cmd=run.pl
beam=15.0
max_active=7000
min_active=200
ivector_scale=1.0
lattice_beam=8.0 # Beam we use in lattice generation.
iter=final
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
parallel_opts=  # ignored now.
scoring_opts=
skip_scoring=false
feat_type=
online_ivector_dir=
minimize=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
  echo " e.g.: $0 --transform-dir exp/tri3b/decode_dev93_tgpr \\"
  echo "      exp/tri3b/graph_tgpr data/test_dev93 exp/tri4a_nnet/decode_dev93_tgpr"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 15.0"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  echo "  --scoring-opts <string>                  # options to local/score.sh"
  echo "  --num-threads <n>                        # number of threads to use, default 1."
  echo "  --parallel-opts <opts>                   # e.g. '--num-threads 4' if you supply --num-threads 4"
  exit 1;
fi

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
model=$srcdir/$iter.mdl


[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $graphdir/HCLG.fst $data/feats.scp $model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
if [ -z "$feat_type" ]; then
  if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=raw; fi
  echo "$0: feature type is $feat_type"
fi

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
  if [ -f $srcdir/delta_order ]; then
    delta_order=`cat $srcdir/delta_order 2>/dev/null`
    feats="$feats add-deltas --delta-order=$delta_order ark:- ark:- |"
  fi
    ;;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)

  if [ $feat_type == "raw" ]; then trans=raw_trans;
  else trans=trans; fi
  if [ $feat_type == "lda" ] && \
    ! cmp $transform_dir/../final.mat $srcdir/final.mat && \
    ! cmp $transform_dir/final.mat $srcdir/final.mat; then
    echo "$0: LDA transforms differ between $srcdir and $transform_dir"
    exit 1;
  fi
  if [ ! -f $transform_dir/$trans.1 ]; then
    echo "$0: expected $transform_dir/$trans.1 to exist (--transform-dir option)"
    exit 1;
  fi
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    for n in $(seq $nj_orig); do cat $transform_dir/$trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/$trans.ark,$dir/$trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/$trans.scp ark:- ark:- |"
  else
    # number of jobs matches with alignment dir.
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/$trans.JOB ark:- ark:- |"
  fi
elif grep 'transform-feats --utt2spk' $srcdir/log/train.1.log >&/dev/null; then
  echo "$0: **WARNING**: you seem to be using a neural net system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi
##

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  # note: subsample-feats, with negative n, will repeat each feature -n times.
  feats="$feats paste-feats --length-tolerance=$ivector_period ark:- 'ark,s,cs:utils/filter_scp.pl $sdata/JOB/utt2spk $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- | copy-matrix --scale=$ivector_scale ark:- ark:-|' ark:- |"
fi

if [ $stage -le 1 ]; then
  $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log \
    nnet-latgen-faster$thread_string \
     --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam \
     --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
     --word-symbol-table=$graphdir/words.txt "$model" \
     $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

if [ $stage -le 2 ]; then
  [ ! -z $iter ] && iter_opt="--iter $iter"
  steps/diagnostic/analyze_lats.sh --cmd "$cmd" $iter_opt $graphdir $dir
fi

# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.

if [ $stage -le 3 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    [ "$iter" != "final" ] && iter_opt="--iter $iter"
    local/score.sh $iter_opt $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi
echo "Decoding done."
exit 0;
