#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


. ./path.sh
. ./cmd.sh

# Training options
backend=pytorch
stage=0
ngpu=0
debugmode=1
dumpdir=dump
N=0
verbose=0
resume=
seed=1
batchsize=20
maxlen_in=800
maxlen_out=150
epochs=15
tag=""
adapt_langs_fn=""

# Feature options
do_delta=false

# Encoder
etype=vggblstmp
elayers=4
eunits=768
eprojs=768
subsample=1_2_2_1_1

# Attention 
atype=location
adim=768
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# Decoder
dlayers=1
dunits=768

# Objective
mtlalpha=0.5
lsm_type=unigram
lsm_weight=0.05
samp_prob=0.0

# Optimizer
opt=adadelta

# Decoding
beam_size=20
nbest=1
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
use_lm=false
decode_nj=32

. ./utils/parse_options.sh || exit 1;

datasets=/export/b15/oadams/datasets-CMU_Wilderness

all_eval_langs_fn=conf/langs/eval_langs
eval_readings_fn=conf/langs/eval_readings

all_eval_langs_train="`basename ${all_eval_langs_fn}`_train"

train_langs="aymara-notgt aymara indonesian-notgt indonesian"

if [ $stage -le 0 ]; then

    # Preparing audio data for each evaluation language and train/dev/test
    # splits for monolingual training
    for reading in `cat ${all_eval_langs_fn} | tr "\n" " "`; do
        echo $reading
        reading_fn="conf/langs/${reading}"
        ./local/prepare_audio_data.sh --langs ${reading_fn} ${datasets}
        ./local/create_splits.sh data/local ${reading_fn} ${reading_fn} ${reading_fn} 
    done

    # Preparing data for each language (a group of readings).
    for train_lang in ${train_langs}; do
        echo $train_lang_fn
        train_lang_fn="conf/langs/${train_lang}"
        ./local/prepare_audio_data.sh --langs ${train_lang_fn} ${datasets}
        ./local/create_splits.sh data/local ${train_lang_fn} ${train_lang_fn} ${train_lang_fn} 
    done

    # Prepare data for all possible evaluation readings that a seed model might
    # get adapted to, so that we can have in advance a dictionary that covers
    # all the languages' graphemes.
    ./local/prepare_audio_data.sh --langs ${all_eval_langs_fn} ${datasets}
    ./local/create_splits.sh data/local ${all_eval_langs_fn} ${all_eval_langs_fn} ${all_eval_langs_fn}
fi

function prepare_dict {

    train_set=$1

    dict=data/lang_1char/${train_set}_units.txt
    nlsyms=data/lang_1char/${train_set}_non_lang_syms.txt

    echo "Dictionary: ${dict}"

    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cat data/${train_set}/text data/${all_eval_langs_train}/text | cut -f 2- | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/${train_set}/text data/${all_eval_langs_train}/text | text2token.py -s 1 -n 1 | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

}

# splits for monolingual training
for reading in `cat ${all_eval_langs_fn} | tr "\n" " "`; do
    echo $reading
    reading_fn="conf/langs/${reading}"
    prepare_dict ${reading}_train
done

# Preparing data for each language (a group of readings).
for train_lang in ${train_langs}; do
    echo $train_lang_fn
    train_lang_fn="conf/langs/${train_lang}"
    prepare_dict ${train_lang}_train
done

prepare_dict ${all_eval_langs_train}
