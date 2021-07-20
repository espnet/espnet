#!/bin/bash

. ./path.sh || die "path.sh expected";

# Remove utt_ids from from training text and add <s> and </s> for SRILM
cut -d' ' -f 2- < data/train/text | awk '{print $0 " </s>"}' \
| awk '{print "<s> " $0}' > data/local/lm_train.txt

# Create .arpa file for trigram LM with Kneser-Ney discounting

ngram-count -text data/local/lm_train.txt -lm data/local/lm.arpa -unk -order 3 -kndiscount3

#convert to FST format for Kaldi
arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang/words.txt \
  data/local/lm.arpa data/lang/G.fst
