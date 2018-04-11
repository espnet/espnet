#!/bin/bash
set -e
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
gpu=0         # use 0 when using GPU on slurm/grid engine, otherwise -1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp #vggblstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# loss related
ctctype=chainer
# decoder related
dlayers=1
dunits=300
# attention related
atype=dot #location
adim=320
awin=5
aheads=1
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.0 #0.5

# label smoothing
lsm_type=unigram
lsm_weight=0.0 #0.05
do_lm=false
# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adam #adadelta
dropout=0.2
epochs=20

# rnnlm related
lm_weight=0.0 #1.0

# decoding parameter
beam_size=5
penalty=0.0
maxlenratio=0.9
minlenratio=0.1
ctc_weight=0.0 #0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# data
dataset='callhome' # options fisher or callhome
task='ST' # options ASR or ST
#wsj0=/export/corpora5/LDC/LDC93S6B
#wsj1=/export/corpora5/LDC/LDC94S13B
# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $dataset == 'fisher' ]; then
  speech_files=/export/a16/gkumar/corpora/LDC2010S01
  if [ $task == 'ST' ]; then
    transcript_files=/export/b07/arenduc1/datasets/fisher-callhome-corpus/en/LDC2010T04
  else
    transcript_files=/export/b07/arenduc1/datasets/fisher-callhome-corpus/LDC2010T04
  fi
  #spanish_lexicon=/export/a16/gkumar/corpora/LDC96L16
  split=local/splits/split_fisher
else
  speech_files=/export/a16/gkumar/corpora/LDC96S35
  if [ $task == 'ST' ]; then
    transcript_files=/export/b07/arenduc1/datasets/fisher-callhome-corpus/en/LDC96T17
  else
    transcript_files=/export/b07/arenduc1/datasets/fisher-callhome-corpus/LDC96T17
  fi
  split=local/splits/split_callhome
fi

dataname=_${task}_${dataset}
#train_set=callhome_train
train_set=train${dataname}
train_dev=dev${dataname}
#train_dev=callhome_dev
train_dev2=dev2${dataname}
#train_test=callhome_test
train_test=test${dataname}
#recog_set="callhome_dev callhome_test"

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    #local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    #local/wsj_format_data.sh
    train_all=data/train_all${dataname}
    if [ ${dataset} == 'fisher' ]; then
      local/fsp_data_prep.sh $speech_files $transcript_files $dataname
      cp -r data/local/data/train_all$dataname $train_all
      splitsets="train dev dev2 test"
    else
      local/callhome_data_prep.sh $speech_files $transcript_files $dataname
      cp -r data/local/data/train_all$dataname $train_all
      splitsets="train dev test"
    fi
    sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" ${train_all}/wav.scp
    local/create_splits.sh --dataname ${dataname} $split "${splitsets}" ${train_all}
    #local/fsp_format_data.sh
    #local/create_splits.sh $split_callhome
fi
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ $dataset == 'fisher' ]; then
  feat_dt2_dir=${dumpdir}/${train_dev2}/delta${do_delta}; mkdir -p ${feat_dt2_dir}
fi
feat_test_dir=${dumpdir}/${train_test}/delta${do_delta}; mkdir -p ${feat_test_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    #for x in train_si284 test_dev93 test_eval92; do
    #for x in train dev dev2 test; do

    sets="$train_dev $train_set $train_test"
    if [ $dataset == 'fisher' ]; then 
      sets="$sets $train_dev2"
    fi
    echo $sets "in stage 1 feature generation"
    #for x in ${train_dev} ${train_set} ${train_test}; do
    for x in $sets; do  
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 50 data/${x} exp/make_fbank/${x} ${fbankdir}
        if [ ${dataset} == 'fisher' ]; then
          remove_longshortdata.sh --maxframes 2000 data/${x} data/${x}.tmp
          [ -d data/${x} ] && \rm -r data/${x}
          mv data/${x}.tmp data/${x}
        fi
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    ## dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{15,18,17,16}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{15,18,17,16}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    if [ $dataset == 'fisher' ]; then
      dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
          data/${train_dev2}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt2_dir}
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_test}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_test_dir}
    #echo "fixing and backing up data"    
    #for f in ${train_set} ${train_dev} ${train_test}; do
    #  utils/fix_data_dir.sh data/${f}
    #  utils/copy_data_dir.sh data/${f} data/${f}.bak
    #  cp ${dumpdir}/${f}/delta${do_delta}/feats.scp data/${f}
    #  utils/fix_data_dir.sh data/${f}
    #  local/word2char.py -t data/${f}/text
    #  mv data/${f}/text.char data/${f}/text
    #done

fi
dict=data/lang_1char${dataname}/${train_set}_units.txt
nlsyms=data/lang_1char${dataname}/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char$dataname/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "[<\[]"  > ${nlsyms}
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
# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs2048
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ] && [ ${do_lm} == true ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
    zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | grep -v "<" | tr [a-z] [A-Z] \
        | text2token.py -n 1 | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' >> ${lmdatadir}/train_others.txt
    cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --gpu ${gpu} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_ctc${ctctype}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_dropout${dropout}_dataset${dataset}_task${task}
    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} ${expdir}/train.log \
        asr_train.py \
        --gpu ${gpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-feat scp:${feat_tr_dir}/feats.scp \
        --valid-feat scp:${feat_dt_dir}/feats.scp \
        --train-label ${feat_tr_dir}/data.json \
        --valid-label ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --ctc_type ${ctctype} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aheads ${aheads} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --dropout-rate ${dropout} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32


    recog_set="${train_dev} ${train_test}"
    if [ $dataset == 'fisher' ]; then
      recog_set="${recog_set} ${train_dev2}"
    fi
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
        data2json.sh --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data.json

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --gpu ${gpu} \
            --backend ${backend} \
            --recog-feat "$feats" \
            --recog-label ${data}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            #--rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        
        awk '{for(i=0; i<NF-1; i++){printf("%s ", $i)} printf("%s\n", $(NF-1))}' ${expdir}/${decode_dir}/ref.wrd.trn > ${expdir}/${decode_dir}/ref.wrd.mt
        awk '{for(i=0; i<NF-1; i++){printf("%s ", $i)} printf("%s\n", $(NF-1))}' ${expdir}/${decode_dir}/hyp.wrd.trn > ${expdir}/${decode_dir}/hyp.wrd.mt
        ./local/multi_bleu.perl ${expdir}/${decode_dir}/ref.wrd.mt < ${expdir}/${decode_dir}/hyp.wrd.mt > ${expdir}/${decode_dir}/result.wrd.bleu


    ) &
    done
    wait
    echo "Finished"
fi

