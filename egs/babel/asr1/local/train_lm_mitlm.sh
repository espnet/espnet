#!/bin/bash -v 

# Copytight 2013  Telefonica (author: Karel Vesely)
# Copyright 2013  Arnab Ghoshal
#                 Johns Hopkins University (author: Daniel Povey)
# Copyright 2012  Vassil Panayotov

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# To be run from one directory above this script.

# Begin configuration section.
ngram_order=3
# end configuration sections

help_message="Usage: "`basename $0`" [options] <lm-corpus> <dict> <out-dir>
Train language model from the LM training corpus.\n
The lm-corpus is normalized ASCII text file, one sentence per line.\n
options: 
  --help          # print this message and exit
  --heldout-sent  # number of held-out sentences, which are used to measure perplexity
  --ngram-order   # order of ngram LM
";

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  printf "$help_message\n";
  exit 1;
fi

text=$1     # data/local/train/text
lexicon=$2  # data/local/dict/lexicon.txt
dir=$3      # data/local/lm

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done


# Search for the MIT-LM training tool
if [ ! -f "$MAIN_ROOT/tools/mitlm/estimate-ngram" ]; then
  echo "--- Downloading and compiling MITLM toolkit ..."
  mkdir -p $MAIN_ROOT/tools
  #command -v svn >/dev/null 2>&1 ||\
  #  { echo "SVN client is needed but not found" ; exit 1; }
  #svn checkout http://mitlm.googlecode.com/svn/trunk/ $MAIN_ROOT/tools/mitlm-svn
	#cd $MAIN_ROOT/tools/mitlm-svn/

  #switch to use github's version
	command -v git >/dev/null 2>&1 ||\
  { echo "GIT client is needed but not found" ; exit 1; }	
	git clone https://github.com/mitlm/mitlm.git $MAIN_ROOT/tools/mitlm
	cd $MAIN_ROOT/tools/mitlm/
  ./autogen.sh
  ./configure --prefix=`pwd`
  make
  make install
  cd ../..
fi

[ ! -d $dir ] && mkdir -p $dir

# Prepare a LM training corpus from the transcripts _not_ in the test set
#cut -f2- -d' ' < $locdata/test_trans.txt |\
#  sed -e 's:[ ]\+: :g' | sort -u > $loctmp/test_utt.txt
#
#cut -f2- -d' ' < $locdata/train_trans.txt |\
#   sed -e 's:[ ]\+: :g' |\
#   gawk 'NR==FNR{test[$0]; next;} !($0 in test)' \
#     $loctmp/test_utt.txt - | sort -u > $loctmp/corpus.txt

# Prepare a wordlist from lexicon
awk '{print $1}' $lexicon | sort | uniq > $dir/wordlist

echo "--- Estimating the LM ..."
if [ ! -f "$MAIN_ROOT/tools/mitlm/estimate-ngram" ]; then
  echo "estimate-ngram not found! MITLM compilation failed?";
  exit 1;
fi
$MAIN_ROOT/tools/mitlm/estimate-ngram -verbose 2 -order $ngram_order -vocab $dir/wordlist -unk \
  -text $text -smoothing ModKN -write-lm $dir/trn.o${ngram_order}g.kn || exit 1
gzip $dir/trn.o${ngram_order}g.kn

echo "*** Finished building the LM model!"
