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
#
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
# TODO: Fix annoying problem of free-gpu not being seen when preparing data on a* machines.
. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

################### DATA SPECIFIC PARAMETERS ##################################
langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="201"
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
snapshot=""

# exp tag
tag="" # tag for managing experiments.

. ./utils/parse_options.sh

# TODO: NEED TO ADD THE RECOG SET # 
train_set=train_e2e
train_dev=dev_e2e
recog_set=105/data/dev10h.pem


if [ $seq_type = "grapheme" ] && [[ $langs = *"101"* ]]; then
  echo >&2 "WARNING: It is probably unwise to use Cantonese (101) when"
  echo >&2 "         training an end-to-end system producing graphemes."
fi 

if [ $stage -le 1 ]; then
  echo "stage 1: Setting up individual languages"
  ./local/setup_languages.sh --langs "${langs}" --seq-type ${seq_type}\
                             --phn-ali ${phoneme_alignments} --recog "${recog}"
fi

###############################################################################
# TRAINING FOR PHONE ALIGNMENTS
#
# This can take a fairly long time and is counter the spirit of end-to-end
# regonition. As such, we never run this by default. For phonetic experiments
# try to use pretrained models to generate alignments if possible or substitute
# word sequences with phonemes directly.
#
# TODO: Support phonetic expansion of text via lexicon lookup or g2p.  
###############################################################################
if [ $seq_type = "phoneme" ] && [ $stage -le 2 ] && [ -z $phoneme_alignments ]; then  
  echo "stage 2: Generate alignments for phonetic experiments"
  ./local/get_alignments.sh
  phoneme_alignments=data/data.ali.phn
fi

###############################################################################
# Create the train dev splits
###############################################################################
if [ $stage -le 3 ]; then
  echo "stage 3: Create data splits"
  num_utts=`cat data/train/text | wc -l`
  num_utts_dev=$(($num_utts / 10))
  
  trainlink=data/train_e2e/train.text
  devlink=data/dev_e2e/dev.text
  
  
  # Deterministic "Random" Split
  python -c "import random; random.seed(1); a=open('data/train/text').readlines(); random.shuffle(a); print((''.join(a[0:${num_utts_dev}])).strip())" | sort > data/dev_e2e.list
  awk '(NR==FNR) {a[$1]=$0; next} !($1 in a){print $1}' data/dev_e2e.list data/train/text | sort > data/train_e2e.list
  ./utils/subset_data_dir.sh --utt-list data/dev_e2e.list data/train data/dev_e2e
  ./utils/subset_data_dir.sh --utt-list data/train_e2e.list data/train data/train_e2e 
  
  [ ! -f $trainlink ] && mv data/train_e2e/text $trainlink
  [ ! -f $devlink ] && mv data/dev_e2e/text $devlink

  if [ ${seq_type} = "phoneme" ]; then
    awk '(NR==FNR) {a[$1]=$0; next} ($1 in a){print a[$1]}' $phoneme_alignments data/train_e2e.list \
      > data/train_e2e/text.phn
    awk '(NR==FNR) {a[$1]=$0; next} ($1 in a){print a[$1]}' $phoneme_alignments data/dev_e2e.list \
      > data/dev_e2e/text.phn
  
    trainlink=data/train_e2e/text.phn
    devlink=data/dev_e2e/text.phn
  fi

  ln -sf ${PWD}/${trainlink} ${PWD}/data/train_e2e/text
  ln -sf ${PWD}/${devlink} ${PWD}/data/dev_e2e/text

  # When using forced alignments, some of the utterances may have failed to
  # align and we have to remove these from the train and dev sets
  ./utils/fix_data_dir.sh data/train_e2e
  ./utils/fix_data_dir.sh data/dev_e2e

fi

# Extract features
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 4 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 4: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
# TODO: Add in test data options and feature generation
    for x in ${train_set} ${train_dev}; do
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

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 5 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 5: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    if [ ! -f data/${train_set}/text ]; then
      echo >&2 "ERROR: File data/${train_set}/text does not exiset."
      echo >&2 "       Please rerun stage 4 to (re)generate this file." 
    fi
    cat data/${train_set}/text |\
      cut -d' ' -f2- | tr " " "\n" | sort | uniq | grep "<.*>" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    if [ $seq_type = "grapheme" ]; then
      text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text |\
        cut -f 2- -d" " | tr " " "\n" | sort | uniq |\
        grep -v -e '^\s*$' |\
        awk '{print $0 " " NR+1}' >> ${dict}
        wc -l ${dict}
    else
      # Use symbols from train
      cat data/${train_set}/text | cut -d' ' -f2- |\
        tr " " "\n" | sort -u | grep -v '^\s*$' |\
        awk '{print $0" "NR+1}' >> ${dict}
        wc -l ${dict}
    fi

    echo "make json files"
    data2json_cmd=data2json.sh
    [ $seq_type = "phoneme" ] && data2json_cmd=local/data2jason_phn.sh
    
    $data2json_cmd --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
        data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    $data2json_cmd --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data.json

fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}


# switch backend
if [[ ${backend} == chainer ]]; then
    train_script=asr_train.py
    decode_script=asr_recog.py
else
    train_script=asr_train_th.py
    decode_script=asr_recog_th.py
fi

if [ ${stage} -le 6 ]; then
    echo "stage 6: Network Training"

    ${cuda_cmd} ${expdir}/train.log \
        ${train_script} \
        --gpu ${gpu} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --train-feat scp:${feat_tr_dir}/feats.scp \
        --valid-feat scp:${feat_dt_dir}/feats.scp \
        --train-label ${feat_tr_dir}/data.json \
        --valid-label ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs} \
        --resume ${snapshot}
fi


if [ ${stage} -le 7 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}

        # split data
        data=data/${rtask}
        split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true data/${train_set}/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
        feats="$feats add-deltas ark:- ark:- |"
        fi

        # make json labels for recognition
        data2json_cmd=data2json.sh
        [ $seq_type = "phoneme" ] && data2json_cmd=local/data2jason_phn.sh

        $data2json_cmd --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data.json

        #### use CPU for decoding
        gpu=-1

        
        # --rnnlm ${lmexpdir}/rnnlm.model.best \
        # --lm-weight ${lm_weight} &
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            ${decode_script} \
            --gpu ${gpu} \
            --recog-feat "$feats" \
            --recog-label ${data}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} &
        wait

        #score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
            
    ) &
    done
    wait
    echo "Finished"
fi


