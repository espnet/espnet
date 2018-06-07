#!/bin/bash
# Copyright 2017  Hossein Hadian
# Apache 2.0

# To be run from ..
# Flat start chain model training.

# This script initializes a trivial tree and transition model
# for flat-start chain training. It then generates the training
# graphs for the training data.

# Begin configuration section.
cmd=run.pl
nj=4
stage=0
shared_phones=true
treedir=              # if specified, the tree and model will be copied from there
                      # note that it may not be flat start anymore.
type=mono             # can be either mono or biphone -- either way
                      # the resulting tree is full (i.e. it doesn't do any tying)

scale_opts="--transition-scale=0.0 --self-loop-scale=0.0"
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: steps/prepare_e2e.sh [options] <data-dir> <lang-dir> <exp-dir>"
  echo " e.g.: steps/prepare_e2e.sh data/train data/lang_chain exp/chain/e2e_tree"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --type <mono | biphone>                          # context dependency type"
  exit 1;
fi

data=$1
lang=$2
dir=$3

if [[ "$type" != "mono" && "$type" != "biphone" ]]; then
  echo "'type' should be either mono or biphone."
  exit 1;
fi

oov_sym=`cat $lang/oov.int` || exit 1;

mkdir -p $dir/log

echo $scale_opts > $dir/scale_opts  # just for easier reference (it is in the logs too)
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $lang/phones.txt $dir || exit 1;

[ ! -f $lang/phones/sets.int ] && exit 1;

if $shared_phones; then
  shared_phones_opt="--shared-phones=$lang/phones/sets.int"
fi

if [ $stage -le 0 ]; then
  if [ -z $treedir ]; then
    echo "$0: Initializing $type system."
    # feat dim does not matter here. Just set it to 10
    $cmd $dir/log/init_${type}_mdl_tree.log \
         gmm-init-$type $shared_phones_opt $lang/topo 10 \
         $dir/0.mdl $dir/tree || exit 1;
  else
    echo "$0: Copied tree/mdl from $treedir." >$dir/log/init_mdl_tree.log
    cp $treedir/final.mdl $dir/0.mdl || exit 1;
    cp $treedir/tree $dir || exit 1;
  fi
  copy-transition-model $dir/0.mdl $dir/0.trans_mdl
  ln -s 0.mdl $dir/final.mdl  # for consistency with scripts which require a final.mdl
fi

lex=$lang/L.fst
if [ $stage -le 1 ]; then
  echo "$0: Compiling training graphs"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
    $dir/tree $dir/0.mdl $lex \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    "ark,scp:$dir/fst.JOB.ark,$dir/fst.JOB.scp" || exit 1;
fi

echo "$0: Done"
