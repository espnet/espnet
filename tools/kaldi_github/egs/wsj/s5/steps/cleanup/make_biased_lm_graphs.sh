#!/bin/bash
# Copyright 2012-2016     Johns Hopkins University (Author: Daniel Povey)
#                2016     Vimal Manohar
# Apache 2.0


# This script creates biased decoding graphs based on the data transcripts as
# HCLG.fsts.scp, in the specified directory; this can be consumed by
# decode_segmentation.sh.
# This is for use in data-cleanup and data-filtering.


set -u
set -o pipefail
set -e

# Begin configuration section.
nj=10
cmd=run.pl
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
top_n_words=100 # Number of common words that we compile into each graph (most frequent
                # in $data/text.orig.
top_n_words_weight=1.0  # this weight is before renormalization; it can be more
                        # or less than 1.
min_words_per_graph=100  # Utterances will be grouped so that they have at least
                         # this many words, before making the graph.
stage=0

### options for make_one_biased_lm.py.
ngram_order=4  # maximum n-gram order to use (but see also --min-lm-state-cout).
min_lm_state_count=10  # make this smaller (e.g. 2) for more strongly biased LM.
discounting_constant=0.3  # strictly between 0 and 1.  Make this closer to 0 for
                          # more strongly biased LM.

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: $0 <data-dir|text> <lang-dir> <dir> <graph-dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri3_cleanup exp/tri3_cleanup/graphs"
   echo "  This script creates biased decoding graphs per utterance (or possibly"
   echo "  groups of utterances, depending on --min-words-per-graph).  Its output"
   echo "  goes to <dir>/HCLG.fsts.scp, indexed by utterance.  Directory <dir> is"
   echo "  required to be a model or alignment directory, containing 'tree' and 'final.mdl'."
   echo "Main options (for others, see top of script file)"
   echo "  --scale-opts <scale-opts>                 # Options relating to language"
   echo "                                            # model scale; default is "
   echo "                                            # '--transition-scale=1.0 --self-loop-scale=0.1'"
   echo "  --top-n-words <N>                         # Number of most-common-words to add with"
   echo "                                            # unigram probabilities into graph (default: 100)"
   echo "  --top-n-words-weight <float>              # Weight given to top-n-words portion of graph"
   echo "                                            # (before renormalizing); may be any positive"
   echo "                                            # number (default: 1.0)"
   echo "  --min-words-per-graph <N>                 # A constant that controls grouping of utterances"
   echo "                                            # (we make the LMs for groups of utterances)."
   echo "                                            # Default: 100."
   echo "  --ngram-order <N>                         # N-gram order in range [2,7].  Maximum n-gram order "
   echo "                                            # that may be used (but also see --min-lm-state-count)."
   echo "                                            # Default 4"
   echo "  --min-lm-state-count <N>                  # Minimum state count for an LM-state of order >2 to "
   echo "                                            # be completely pruned away [bigrams will always be kept]"
   echo "                                            # Default 10.  Smaller -> more strongly biased LM"
   echo "  --discounting-constant <float>            # Discounting constant for Kneser-Ney, strictly between 0"
   echo "                                            # and 1.  Default 0.3.  Smaller -> more strongly biased LM."
   echo "  --config <config-file>                    # config containing options"
   echo "  --nj <nj>                                 # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data_or_text=$1
lang=$2
dir=$3
graph_dir=$4

if [ -d $data_or_text ]; then
  text=$data_or_text/text
else
  text=$data_or_text
fi

mkdir -p $graph_dir

for f in $text $lang/oov.int $dir/tree $dir/final.mdl \
    $lang/L_disambig.fst $lang/phones/disambig.int; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

utils/lang/check_phones_compatible.sh $lang/phones.txt $dir/phones.txt
cp $lang/phones.txt $graph_dir

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $graph_dir/log

# create top_words.{int,txt}
if [ $stage -le 0 ]; then
  export LC_ALL=C
  # the following pipe will be broken due to the 'head'; don't fail.
  set +o pipefail
  utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $text | \
    awk '{for(x=2;x<=NF;x++) print $x;}' | sort | uniq -c | \
     sort -nr | head -n $top_n_words > $graph_dir/word_counts.int
  set -o pipefail
  total_count=$(awk '{x+=$1} END{print x}' < $graph_dir/word_counts.int)
  # print top-n words with their unigram probabilities.
  awk -v tot=$total_count -v weight=$top_n_words_weight '{print $2, ($1*weight)/tot;}' \
     <$graph_dir/word_counts.int >$graph_dir/top_words.int
  utils/int2sym.pl -f 1 $lang/words.txt <$graph_dir/top_words.int >$graph_dir/top_words.txt
fi

word_disambig_symbol=$(cat $lang/words.txt | grep -w "#0" | awk '{print $2}')
if [ -z "$word_disambig_symbol" ]; then
  echo "$0: error getting word disambiguation symbol"
  exit 1
fi

mkdir -p $graph_dir/texts
split_text=
for n in `seq $nj`; do
  split_text="$split_text $graph_dir/texts/text.$n"
done

utils/split_scp.pl $text $split_text

mkdir -p $graph_dir/log $graph_dir/fsts

# Make $dir an absolute pathname
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

if [ $stage -le 1 ]; then
  echo "$0: creating utterance-group-specific decoding graphs with biased LMs"

  # These options are passed through directly to make_one_biased_lm.py.
  lm_opts="--word-disambig-symbol=$word_disambig_symbol --ngram-order=$ngram_order --min-lm-state-count=$min_lm_state_count --discounting-constant=$discounting_constant --top-words=$graph_dir/top_words.int"

  $cmd JOB=1:$nj $graph_dir/log/compile_decoding_graphs.JOB.log \
    utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $graph_dir/texts/text.JOB \| \
    steps/cleanup/make_biased_lms.py --min-words-per-graph=$min_words_per_graph \
      --lm-opts="$lm_opts" $graph_dir/fsts/utt2group.JOB \| \
    compile-train-graphs-fsts $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
      $dir/tree $dir/final.mdl $lang/L_disambig.fst ark:- \
    ark,scp:$graph_dir/fsts/HCLG.fsts.JOB.ark,$graph_dir/fsts/HCLG.fsts.JOB.scp || exit 1
fi

for j in $(seq $nj); do cat $graph_dir/fsts/HCLG.fsts.$j.scp; done > $graph_dir/fsts/HCLG.fsts.per_utt.scp
for j in $(seq $nj); do cat $graph_dir/fsts/utt2group.$j; done > $graph_dir/fsts/utt2group


cp $lang/words.txt $graph_dir/
cp -r $lang/phones $graph_dir/

# The following command gives us an scp file relative to utterance-id.
utils/apply_map.pl -f 2 $graph_dir/fsts/HCLG.fsts.per_utt.scp <$graph_dir/fsts/utt2group > $graph_dir/HCLG.fsts.scp

n1=$(cat $text | wc -l)
n2=$(cat $graph_dir/HCLG.fsts.scp | wc -l)

if [ $[$n1*9] -gt $[$n2*10] ]; then
  echo "$0: too many utterances have no scp, something seems to have gone wrong."
  exit 1
fi

exit 0;
