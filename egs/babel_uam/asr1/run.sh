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
phoneme_alignments=


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
train_set=train
train_dev=dev
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
    else
      ./local/prepare_data.sh
    fi
    
    ###########################################################################
    # Create dictionaries with split diphthongs and standardized tones
    ###########################################################################
    # In the lexicons provided by babel there are phonemes x_y, for which _y may
    # or may not best be considered as a tag on phoneme x. In Lithuanian, for
    # instance, there is a phoneme A_F for which _F or indicates failling tone.
    # This same linguistic feature is represented in other languages as a "tag"
    # (i.e. 判 pun3 p u: n _3), which means for the purposes of kaldi, that
    # those phonemes share a root in the clustering decision tree, and the tag
    # becomes an extra question. We may want to revisit this issue later.
    echo "Dictionary ${l}"
    dict=data/dict_universal
    
    mkdir -p $dict
    # Create silence lexicon
    echo -e "<silence>\tSIL\n<unk>\t<oov>\n<noise>\t<sss>\n<v-noise>\t<vns>" \
      > ${dict}/silence_lexicon.txt
    
    # Create non-silence lexicon
    grep -vFf ${dict}/silence_lexicon.txt data/local/lexicon.txt \
      > data/local/nonsilence_lexicon.txt
    
    # Create split diphthong and standarized tone lexicons for nonsilence words
    ./local/prepare_universal_lexicon.py \
      ${dict}/nonsilence_lexicon.txt data/local/nonsilence_lexicon.txt \
      local/phone_maps/${l} 
    
    cat ${dict}/{,non}silence_lexicon.txt | sort > ${dict}/lexicon.txt
    
    # Prepare the rest of the dictionary directory
    # -----------------------------------------------
    # The local/prepare_dict.py script, which is basically the same as
    # prepare_unicode_lexicon.py used in the babel recipe to create the
    # graphemic lexicons, is better suited for working with kaldi formatted
    # lexicons and can be used for this task by only modifying optional input
    # arguments. If we could modify local/prepare_lexicon.pl to accomodate this
    # need it may be more intuitive.
    ./local/prepare_dict.py \
      --silence-lexicon ${dict}/silence_lexicon.txt ${dict}/lexicon.txt ${dict}

    ###########################################################################
    # Prepend language ID to all utterances to disambiguate between speakers
    # of different languages sharing the same speaker id.
    #
    # The individual lang directories can be used for alignments, while a
    # combined directory will be used for training. This probably has minimal
    # impact on performance as only words repeated across languages will pose
    # problems and even amongst these, the main concern is the <hes> marker.
    ###########################################################################
    echo "Prepend ${l} to data dir"
    ./utils/copy_data_dir.sh --spk-prefix "${l}_" --utt-prefix "${l}_" \
      data/train data/train_${l}
      cd $cwd
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
    dict_dirs="data/${l}/data/dict_universal ${dict_dirs}"
  done

  ./utils/combine_data.sh data/train $train_dirs

  # This script was made to mimic the utils/combine_data.sh script, but instead
  # it merges the lexicons while reconciling the nonsilence_phones.txt,
  # silence_phones.txt, and extra_questions.txt by basically just calling
  # local/prepare_unicode_lexicon.py. As mentioned, it may be better to simply
  # modify an existing script to automatically create the dictionary dir from
  # a lexicon, rather than overuse the local/prepare_unicode_lexicon.py script.
  ./local/combine_lexicons.sh data/dict_universal $dict_dirs

  # Prepare lang directory
  ./utils/prepare_lang.sh --share-silence-phones true \
    data/dict_universal "<unk>" data/dict_universal/tmp.lang data/lang_universal
fi


###############################################################################
# TRAINING FOR PHONE ALIGNMENTS
###############################################################################
if [ $seq_type = "phoneme" ] && [ $stage -le 3 ]; then  
  if [ ! -f exp/mono/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Starting (small) monophone training in exp/mono on" `date`
    echo ---------------------------------------------------------------------
    steps/train_mono.sh \
      --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
      data/train_sub1 data/lang_universal exp/mono
    touch exp/mono/.done
  fi


  if [ ! -f exp/tri1/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Starting (small) triphone training in exp/tri1 on" `date`
    echo ---------------------------------------------------------------------
    steps/align_si.sh \
      --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
      data/train_sub2 data/lang_universal exp/mono exp/mono_ali_sub2

    steps/train_deltas.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
      data/train_sub2 data/lang_universal exp/mono_ali_sub2 exp/tri1

    touch exp/tri1/.done
  fi

  echo ---------------------------------------------------------------------
  echo "Starting (medium) triphone training in exp/tri2 on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -f exp/tri2/.done ]; then
    steps/align_si.sh \
      --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
      data/train_sub3 data/lang_universal exp/tri1 exp/tri1_ali_sub3

    steps/train_deltas.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
      data/train_sub3 data/lang_universal exp/tri1_ali_sub3 exp/tri2

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train_sub3 data/lang_universal data/dict_universal \
      exp/tri2 data/dict_universal/dictp/tri2 data/dict_universal/langp/tri2 data/lang_universalp/tri2

    touch exp/tri2/.done
  fi

  echo ---------------------------------------------------------------------
  echo "Starting (full) triphone training in exp/tri3 on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -f exp/tri3/.done ]; then
    steps/align_si.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang_universalp/tri2 exp/tri2 exp/tri2_ali

    steps/train_deltas.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" \
      $numLeavesTri3 $numGaussTri3 data/train data/lang_universalp/tri2 exp/tri2_ali exp/tri3

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train data/lang_universal data/dict_universal/ \
      exp/tri3 data/dict_universal/dictp/tri3 data/dict_universal/langp/tri3 data/lang_universalp/tri3

    touch exp/tri3/.done
  fi


  echo ---------------------------------------------------------------------
  echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -f exp/tri4/.done ]; then
    steps/align_si.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang_universalp/tri3 exp/tri3 exp/tri3_ali

    steps/train_lda_mllt.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" \
      $numLeavesMLLT $numGaussMLLT data/train data/lang_universalp/tri3 exp/tri3_ali exp/tri4

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train data/lang_universal data/dict_universal \
      exp/tri4 data/dict_universal/dictp/tri4 data/dict_universal/langp/tri4 data/lang_universalp/tri4

    touch exp/tri4/.done
  fi

  echo ---------------------------------------------------------------------
  echo "Starting (SAT) triphone training in exp/tri5 on" `date`
  echo ---------------------------------------------------------------------

  if [ ! -f exp/tri5/.done ]; then
    steps/align_si.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang_universalp/tri4 exp/tri4 exp/tri4_ali

    steps/train_sat.sh \
      --boost-silence $boost_sil --cmd "$train_cmd" \
      $numLeavesSAT $numGaussSAT data/train data/lang_universalp/tri4 exp/tri4_ali exp/tri5

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train data/lang_universal data/dict_universal \
      exp/tri5 data/dict_universal/dictp/tri5 data/dict_universal/langp/tri5 data/lang_universalp/tri5

    touch exp/tri5/.done
  fi

  if [ ! -f exp/tri5_ali/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Starting exp/tri5_ali on" `date`
    echo ---------------------------------------------------------------------
    steps/align_fmllr.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      data/train data/lang_universalp/tri5 exp/tri5 exp/tri5_ali

    local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
      data/train data/lang_universal data/dict_universal \
      exp/tri5_ali data/dict_universal/dictp/tri5_ali data/dict_universal/langp/tri5_ali data/lang_universalp/tri5_ali

    touch exp/tri5_ali/.done
  fi
fi


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train dev dev10h.hat.pem; do
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









