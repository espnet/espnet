#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


###############################################################################
#                      Universal Acoustic Models
###############################################################################
#
# This script sets up data from the BABEL Languages to be used to train
# universal acoustic models. By default, it leaves out 4 languages:
#
#  - 201_haitian: Many results numbers with which to compare for this language.
#  - 307_amharic: Left out to repeat LORELEI experiments.
#  - 107_vietnamese: Left out to test performance on a tonal language.
#  - 404_georgian: The final evaluation language from BABEL.
#
# which are used to test the trained universal acoustic models. The script
# consists of the following steps:
#   1. Prepare data directories
#   2. Standardize the lexicons
#   3. Training
#
# We can train either directly on the grapheme sequences or on the phoneme
# sequences. A flag 
#        seq_type="grapheme"
#        seq_type="phoneme"
# 
# sets this option. To train on the phoneme sequences requires getting phoneme
# aligned data, which requires training a standard HMM/GMM system and force
# aligning the training text and training audio. This training stage can take a
# fair amount of time. I've provided the (for the JHU folks) the generated
# alignment directly as a text file as in kaldi, to bypass this step for now. In
# fact with that file you can skip most of the script as it is just there to
# prepare the data needed to generate the attached file.
#
#
###############################################################################

set -e
set -o pipefail
. ./path.sh
. ./cmd.sh
. ./conf/common_vars.sh

################### DATA SPECIFIC PARAMETERS ##################################
langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306
       401 402 403"
seq_type="grapheme"
stage=0
phoneme_alignments=/export/a15/MStuDy/Matthew/LORELEI/kaldi/egs/universal_acoustic_model/s5_all_babel_llp/data/train/text.ali.phn

######################### E2E PARAMATERS ######################################
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
gpu=-1         # use 0 when using GPU on slurm/grid engine, otherwise -1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
lm_weight=1.0

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# exp tag
tag="" # tag for managing experiments.

. ./utils/parse_options.sh


### TODO NEED TO ADD THE RECOG SET #### 
train_set=train_e2e
train_dev=dev_e2e
recog_set=dev10h.hat.pem

cwd=$(utils/make_absolute.sh `pwd`)
if [ $stage -le 0 ]; then
  # Create a language specific directory for each language
  for l in ${langs}; do
    [ -d data/${l} ] || mkdir -p data/${l}
    cd data/${l}

    # Copy the main directories from the top level into
    # each language specific directory
    ln -sf ${cwd}/local .
    for f in ${cwd}/{utils,steps,conf}; do
      link=`make_absolute.sh $f`
      ln -sf $link .
    done

    conf_file=`find conf/lang -name "${l}-*limitedLP*.conf" \
                           -o -name "${l}-*LLP*.conf" | head -1`

    echo "----------------------------------------------------"
    echo "Using language configurations: ${conf_file}"
    echo "----------------------------------------------------"

    cp ${conf_file} lang.conf
    cp ${cwd}/cmd.sh .
    cp ${cwd}/path_babel.sh path.sh
    cd ${cwd}
  done
fi


for l in ${langs}; do
  cd data/${l}
  #############################################################################
  # Prepare the data directories (train and dev10h) directories
  #############################################################################
  if [ $stage -le 1 ]; then
    if [ $seq_type = "phoneme" ]; then
      ./local/prepare_data.sh --extract-feats true
      ./local/prepare_universal_dict.sh --dict data/dict_universal ${l} 
    else
      ./local/prepare_data.sh
    fi
  fi
  cd ${cwd}
done

###############################################################################
# Combine all langauge specific training directories and generate a single
# lang directory by combining all langauge specific dictionaries
###############################################################################
if [ $stage -le 2 ]; then
  train_dirs=""
  dict_dirs=""
  for l in ${langs}; do
    train_dirs="data/${l}/data/train_${l} ${train_dirs}"
    if [ $seq_type = "phoneme" ]; then
      dict_dirs="data/${l}/data/dict_universal ${dict_dirs}"
    fi
  done

  ./utils/combine_data.sh data/train $train_dirs

  if [ $seq_type = "phoneme" ]; then
    ./local/combine_lexicons.sh data/dict_universal $dict_dirs
    # Prepare lang directory
    ./utils/prepare_lang.sh --share-silence-phones true \
      data/dict_universal "<unk>" data/dict_universal/tmp.lang data/lang_universal
  fi
fi

###############################################################################
# TRAINING FOR PHONE ALIGNMENTS
###############################################################################
if [ $seq_type = "phoneme" ] && [ $stage -le 3 ] && [ -z $phoneme_aligmnets ]; then  
  # TODO Add the ali-to-phones part to local/get_alignments.sh
  ./local/get_alignments.sh
  phoneme_alignments=data/data.ali.phn
fi

###############################################################################
# Create the train dev splits
###############################################################################
if [ $stage -le 4 ]; then
  num_utts=`cat data/train/text | wc -l`
  num_utts_dev=$(($num_utts / 10))
  
  trainlink=data/train_e2e/text
  devlink=data/dev_e2e/text
  
  
  # Deterministic "Random" Split
  python -c "import random; random.seed(1); a=open('data/train/text').readlines(); random.shuffle(a); print((''.join(a[0:${num_utts_dev}])).strip())" | sort > data/dev_e2e.list
  awk '(NR==FNR) {a[$1]=$0; next} !($1 in a){print $1}' data/dev_e2e.list data/train/text | sort > data/train_e2e.list
  ./utils/subset_data_dir.sh --utt-list data/dev_e2e.list data/train data/dev_e2e
  ./utils/subset_data_dir.sh --utt-list data/train_e2e.list data/train data/train_e2e 
  
  
  if [ ${seq_type} = "phoneme" ]; then
    awk '(NR==FNR) {a[$1]=$0; next} ($1 in a){print a[$1]}' $phoneme_alignments data/train_e2e.list \
      > data/train_e2e/text.phn
    awk '(NR==FNR) {a[$1]=$0; next} ($1 in a){print a[$1]}' $phoneme_alignments data/dev_e2e.list \
      > data/dev_e2e/text.phn
  
    trainlink=data/train_e2e/text.phn
    devlink=data/dev_e2e/text.phn
  fi

  ln -sF ${PWD}/${trainlink} ${PWD}/data/train_e2e/train.text
  ln -sF ${PWD}/${devlink} ${PWD}/data/dev_e2e/dev.text
fi

# Extract features
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 5 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train_e2e dev_e2e dev10h.hat.pem; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
fi

exit
dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
fi


exit 0;





