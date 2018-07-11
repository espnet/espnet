#!/bin/bash

train_set=traindev
recog_set="eval_102"

stage=0
stage_train=0
nj_train=20
stage_last=1000000

lang_train=data/lang.wrd2grp # will be created

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# ----- Data
if [ $stage -le 0 ] && [ $stage_last -gt 0 ]; then
    [ ! -d data/${train_set} ] && ./utils/combine_data.sh data/${train_set} data/train data/dev
fi    

# ----- Make grapheme based dct+lang
dctdir_tmp=data/local/dict.tmp
if [ $stage -le 1 ] && [ $stage_last -gt 1 ]; then
    # Create dictionary into ${lang_train}/dct
    ./local/make_dct.grp.sh data/${train_set} ${lang_train}

    # Create gmm ${lang_train} from dct in ${lang_train}/dct ($1)
    ./local/make_lang.gmm.sh ${lang_train} ${dctdir_tmp} ${lang_train} 

fi

# ------ Make LM
if [ $stage -le 4 ] && [ $stage_last -gt 4 ]; then
    
    mkdir -p data/local/lm.$train_set
    #    awk '{$1=""; print}' data/$train_set/text | sed 's:^[[:blank:]]\+::;s:<sil>::g;s:<unk>::g' > data/local/lm.nounk.$train_set/text  # nounk
    awk '{$1=""; print}' data/$train_set/text | sed 's:^[[:blank:]]\+::;s:<sil>::g' > data/local/lm.$train_set/text
    
    cat ${dctdir_tmp}/lexicon.txt |\
      awk '$1 !~ /^-/ && $1 !~ /-$/{print}' > ${dctdir_tmp}/lexicon.filt.txt
    
    lm_corpus=data/local/lm.$train_set/text
    dct=${dctdir_tmp}/lexicon.txt #filt.txt
    local/train_lm_mitlm.sh --ngram-order 3\
      $lm_corpus $dct $(dirname $lm_corpus) || exit 1
    
    LM=$(dirname $lm_corpus)/trn.o3g.kn.gz
    lang_test=data/lang_test.wrd2grp.$train_set.o3g.kn
    utils/format_lm.sh $lang_train $LM $dct $lang_test || exit 1
fi

# ----- Make fea
feakind=MultRDTv1
fbankdir=data-$feakind

if [ $stage -le 5 ] && [ $stage_last -gt 5 ]; then

    for x in ${train_set} ${recog_set}; do
	local/fea.genMultRDT.sh \
            data/$x data-$feakind/$x
    done
fi

# ---- Gmm training
if [ $stage -le 10 ] && [ $stage_last -gt 10 ]; then

    [ ! -e local/score.sh ] && ln -sr $PWD/steps/score_kaldi.sh local/score.sh
    lang_test=data/lang_test.wrd2grp.$train_set.o3g.kn # from prev stage
    ./local/run-1-gmm.MultRDTv1_short.sh \
	--stage ${stage_train} --nj $nj_train \
	--lang $lang_train \
	--lang_test $lang_test   \
	--train $fbankdir/${train_set} \
	--dev   $fbankdir/${recog_set}
fi
