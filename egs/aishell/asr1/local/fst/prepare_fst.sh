#!/usr/bin/env bash
  
# Copyright 2021 Gaopeng Xu
# Apache 2.0

. ./path.sh
data=/e2e
dict=data/lang_1char/train_sp_units.txt
stage=0
n_gram=4
lm=data/local/lm
if [ ${stage} -le 0 ]; then
    #  Prepare dict
    mkdir -p data/fst
    cp $dict data/fst/units.txt
    local/fst/prepare_dict.py $dict ${data}/resource_aishell/lexicon.txt \
        data/fst/lexicon.txt
    echo "<unk> <unk>" >>data/fst/lexicon.txt
    local/fst/make_L.sh data/fst data/fst/tmp data/fst/lang 
   
    # Train lm
    lm=data/local/lm
    mkdir -p $lm
    text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " > $lm/text
    lmplz --discount_fallback -o ${n_gram} <$lm/text> $lm/${n_gram}gram.arpa
    # Build decoding TLG
    python3 -m kaldilm \
	    --read-symbol-table="data/fst/lang/words.txt" \
	    --disambig-symbol='#0' \
	    --max-order=${n_gram} \
	    $lm/${n_gram}gram.arpa >data/fst/lang/G.fst.txt
fi
