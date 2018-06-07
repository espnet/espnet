#!/bin/bash
# Copyright 2012        Johns Hopkins University (Author: Daniel Povey)
#           2014-2015   Vimal Manohar
# Apache 2.0.

# Create denominator lattices for MMI/MPE training [deprecated].
# This version uses the neural-net models (version 3, i.e. the nnet3 code).
# Creates its output in $dir/lat.*.gz
# Note: the more recent discriminative training scripts will not use this
# script at all, they'll use get_degs.sh which combines the decoding
# and egs-dumping into one script (to save disk space and disk I/O).

# Begin configuration section.
nj=4
cmd=run.pl
sub_split=1
beam=13.0
frames_per_chunk=50
lattice_beam=7.0
self_loop_scale=0.1
acwt=0.1
max_active=5000
min_active=200
max_mem=20000000 # This will stop the processes getting too large.
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
num_threads=1 # number of threads of decoder [only applicable if not looped, for now]
online_ivector_dir=
determinize=true
minimize=false
ivector_scale=1.0
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
# End configuration section.


echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

num_threads=1 # Fixed to 1 for now

if [ $# != 4 ]; then
  echo "Usage: steps/nnet3/make_denlats.sh [options] <data-dir> <lang-dir> <src-dir> <exp-dir>"
  echo "  e.g.: steps/nnet3/make_denlats.sh data/train data/lang exp/nnet4 exp/nnet4_denlats"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --sub-split <n-split>                            # e.g. 40; use this for "
  echo "                           # large databases so your jobs will be smaller and"
  echo "                           # will (individually) finish reasonably soon."
  echo "  --num-threads  <n>                # number of threads per decoding job"
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4


extra_files=
[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
for f in $data/feats.scp $lang/L.fst $srcdir/final.mdl $extra_files; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

sdata=$data/split$nj
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;

oov=`cat $lang/oov.int` || exit 1;

cp -rH $lang $dir/

# Compute grammar FST which corresponds to unigram decoding graph.
new_lang="$dir/"$(basename "$lang")

# mkgraph.sh expects a whole directory "lang", so put everything in one directory...
# it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and
# final.mdl from $srcdir; the output HCLG.fst goes in $dir/graph.

echo "Compiling decoding graph in $dir/dengraph"
if [ -s $dir/dengraph/HCLG.fst ] && [ $dir/dengraph/HCLG.fst -nt $srcdir/final.mdl ]; then
  echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
else
  echo "Making unigram grammar FST in $new_lang"
  cat $data/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
   awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
    utils/make_unigram_grammar.pl | fstcompile | fstarcsort --sort_type=ilabel > $new_lang/G.fst \
    || exit 1;
  utils/mkgraph.sh --self-loop-scale $self_loop_scale $new_lang $srcdir $dir/dengraph || exit 1;
fi
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null

echo "$0: feature type is raw"

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

# if this job is interrupted by the user, we want any background jobs to be
# killed too.
cleanup() {
  local pids=$(jobs -pr)
  [ -n "$pids" ] && kill $pids
}
trap "cleanup" INT QUIT TERM EXIT

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

if [ -f $srcdir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
  cp $srcdir/frame_subsampling_factor $dir
fi

lattice_determinize_cmd=
if $determinize; then
  lattice_determinize_cmd="lattice-determinize-non-compact --acoustic-scale=$acwt --max-mem=$max_mem --minimize=$minimize --prune=true --beam=$lattice_beam ark:- ark:- |"
fi

if [ $sub_split -eq 1 ]; then
  $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode_den.JOB.log \
    nnet3-latgen-faster$thread_string $ivector_opts $frame_subsampling_opt \
    --frames-per-chunk=$frames_per_chunk \
    --extra-left-context=$extra_left_context \
    --extra-right-context=$extra_right_context \
    --extra-left-context-initial=$extra_left_context_initial \
    --extra-right-context-final=$extra_right_context_final \
    --minimize=false --determinize-lattice=false \
    --word-determinize=false --phone-determinize=false \
    --max-active=$max_active --min-active=$min_active --beam=$beam \
    --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=false \
    --max-mem=$max_mem --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
    $dir/dengraph/HCLG.fst "$feats" \
    "ark:|$lattice_determinize_cmd gzip -c >$dir/lat.JOB.gz" || exit 1
else

  # each job from 1 to $nj is split into multiple pieces (sub-split), and we aim
  # to have at most two jobs running at each time.  The idea is that if we have stragglers
  # from one job, we can be processing another one at the same time.
  rm $dir/.error 2>/dev/null

  prev_pid=
  for n in `seq $[nj+1]`; do
    if [ $n -gt $nj ]; then
      this_pid=
    elif [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $srcdir/final.mdl ]; then
      echo "Not processing subset $n as already done (delete $dir/.done.$n if not)";
      this_pid=
    else
      sdata2=$data/split$nj/$n/split${sub_split}utt;
      split_data.sh --per-utt $sdata/$n $sub_split || exit 1;
      mkdir -p $dir/log/$n
      mkdir -p $dir/part
      feats_subset=`echo $feats | sed s:JOB/:$n/split${sub_split}utt/JOB/:g`

      $cmd --num-threads $num_threads JOB=1:$sub_split $dir/log/$n/decode_den.JOB.log \
        nnet3-latgen-faster$thread_string $ivector_opts $frame_subsampling_opt \
        --frames-per-chunk=$frames_per_chunk \
        --extra-left-context=$extra_left_context \
        --extra-right-context=$extra_right_context \
        --extra-left-context-initial=$extra_left_context_initial \
        --extra-right-context-final=$extra_right_context_final \
        --minimize=false --determinize-lattice=false \
        --word-determinize=false --phone-determinize=false \
        --max-active=$max_active --min-active=$min_active --beam=$beam \
        --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=false \
        --max-mem=$max_mem --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
        $dir/dengraph/HCLG.fst "$feats_subset" \
        "ark:|$lattice_determinize_cmd gzip -c >$dir/lat.$n.JOB.gz" || touch $dir/.error &
      this_pid=$!
    fi
    if [ ! -z "$prev_pid" ]; then  # Wait for the previous job; merge the previous set of lattices.
      wait $prev_pid
      [ -f $dir/.error ] && echo "$0: error generating denominator lattices" && exit 1;
      rm $dir/.merge_error 2>/dev/null
      echo Merging archives for data subset $prev_n
      for k in `seq $sub_split`; do
        gunzip -c $dir/lat.$prev_n.$k.gz || touch $dir/.merge_error;
      done | gzip -c > $dir/lat.$prev_n.gz || touch $dir/.merge_error;
      [ -f $dir/.merge_error ] && echo "$0: Merging lattices for subset $prev_n failed (or maybe some other error)" && exit 1;
      rm $dir/lat.$prev_n.*.gz
      touch $dir/.done.$prev_n
    fi
    prev_n=$n
    prev_pid=$this_pid
  done
fi


echo "$0: done generating denominator lattices."
