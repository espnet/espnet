#!/bin/bash
set -eo pipefail
. local/fst/parse_options.sh

if [ $# -ne 3 ]; then
  echo "usage: local/fst/make_L.sh <dict-src-dir> <tmp-dir> <lang-dir>"
  echo "e.g.: local/fst/make_L.sh data/local/dict data/local/lang_tmp data/lang"
  echo "<dict-src-dir> should contain the following files:"
  echo "lexicon.txt lexicon_numbers.txt units.txt"
  echo "options: "
  exit 1;
fi

srcdir=$1
tmpdir=$2
dir=$3
mkdir -p $dir $tmpdir
grammar_opts=
[ -f path.sh ] && . ./path.sh

cp $srcdir/units.txt $dir

# Add probabilities to lexicon entries. There is in fact no point of doing this here since all the entries have 1.0.
# But utils/make_lexicon_fst.pl requires a probabilistic version, so we just leave it as it is.
perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $srcdir/lexicon.txt > $tmpdir/lexiconp.txt || exit 1;

# Add disambiguation symbols to the lexicon. This is necessary for determinizing the composition of L.fst and G.fst.
# Without these symbols, determinization will fail.
ndisambig=`local/fst/add_lex_disambig.pl $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt`
ndisambig=$[$ndisambig+1];

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > $tmpdir/disambig.list

# Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>,
# the actual model unit, and the disambiguation symbols.
cat $srcdir/units.txt | awk '{print $1}' > $tmpdir/units.list
(echo '<eps>';) | cat - $tmpdir/units.list $tmpdir/disambig.list | awk '{print $1 " " (NR-1)}' > $dir/tokens.txt
#i#ised -i 's/<blank>/<eps>/g' $tmpdir/units.list
#sed -i "1i<eps>" $tmpdir/units.list
#cat $tmpdir/units.list $tmpdir/disambig.list | awk '{print $1 " " (NR-1)}' > $dir/tokens.txt
# Encode the words with indices. Will be used in lexicon and language model FST compiling.
cat $tmpdir/lexiconp.txt | awk '{print $1}' | sort | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' > $dir/words.txt || exit 1;

# Now compile the lexicon FST. Depending on the size of your lexicon, it may take some time.
token_disambig_symbol=`grep \#0 $dir/tokens.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $dir/words.txt | awk '{print $2}'`

local/fst/make_lexicon_fst.py $grammar_opts \
	--sil-prob=0 --sil-phone="sil" --sil-disambig='#'$ndisambig \
	$tmpdir/lexiconp_disambig.txt | \
	local/fst/sym2int.pl -f 3 $dir/tokens.txt | \
	local/fst/sym2int.pl -f 4 $dir/words.txt | \
	local/fst/fstaddselfloops.pl $token_disambig_symbol  $word_disambig_symbol > $dir/L_disambig.fst.txt || exit 1;



echo "Lexicon FSTs compiling succeeded"
