#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a task of 10 language-indepent ASR used in
# S. Watanabe et al, "Language independent end-to-end architecture for
# joint language identification and speech recognition," Proc. ASRU'17, pp. 265--269 (2017)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
storage=/mnt/scratch01/tmp/karafiat/espnet/$RANDOM

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# loss related
#ctctype=chainer
# decoder related
dlayers=3 #1
dunits=300
# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=50
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'


train_set=train_mult
#train_set=train_mult
train_dev=dev_mult


# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#train_dev=dev_mult
recog_set="dt_102 dt_106 dt_202 dt_203 dt_206 dt_en dt_csj_japanese et_102 et_106 et_202 et_203 et_206 et_en_1 et_en_2 et_en_3 et_en_4 et_csj_japanese_1 et_csj_japanese_2 et_csj_japanese_3"

# Pre-defined list of Babel languages for train and test
train_babel_langs="101 103 104 105 404 107 201 204 205 207"
target_babel_langs="102 106 202 203 206"
olddir_karthick=/mnt/matylda6/baskar/espnet/egs/
olddir=/mnt/matylda3/karafiat/BABEL/GIT/espnet.github/egs
sourcedir_babel=/mnt/matylda3/karafiat/BABEL/GIT/espnet.github.v2/egs/babel/Baselines.v2

if [ ${stage} -le 0 ]; then
    # TODO
    # add a check whether the following data preparation is completed or not
    # Librespeech
    false  && {
    lang_code=en
    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/librispeech/asr1/data/train_960 data/tr_${lang_code}
    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/librispeech/asr1/data/dev data/dt_${lang_code}
    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/librispeech/asr1/data/test_clean data/et_${lang_code}_1
    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/librispeech/asr1/data/test_other data/et_${lang_code}_2
    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/librispeech/asr1/data/dev_clean data/et_${lang_code}_3
    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/librispeech/asr1/data/dev_other data/et_${lang_code}_4
    }
    # CSJ Japanese (INFO: not available in BUT)
    false && {
    lang_code=csj_japanese
#    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/csj/asr1/data/train_nodup data/tr_${lang_code}
#    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/csj/asr1/data/train_dev data/dt_${lang_code}
#    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/csj/asr1/data/eval1 data/et_${lang_code}_1
#    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/csj/asr1/data/eval2 data/et_${lang_code}_2
#    utils/copy_data_dir.sh --utt-suffix -${lang_code} $olddir/csj/asr1/data/eval3 data/et_${lang_code}_3
#    # 1) change wide to narrow chars
#    # 2)5B lower to upper chars
#    for x in data/*_${lang_code}*; do
#        utils/copy_data_dir.sh ${x} ${x}_org
#        cat ${x}_org/text | nkf -Z |\
#            awk '{for(i=2;i<=NF;++i){$i = toupper($i)} print}' > ${x}/text
#        rm -fr ${x}_org
#    done

    for f in tr_${lang_code} dt_${lang_code} et_${lang_code}_{1,2,3}; do 
	[ ! -e data/$f ] && ln -s $olddir_karthick/babel/asr1/data/$f data/$f
    done

    }

    # 15 BABEL languages (10 for train, 5 for test)
    # This lang code will change for JHU (just modify the data path)
    # Assuming the run.sh --langs "" --recog "" is executed
    for lang_code in $train_babel_langs; do
	utils/copy_data_dir.sh --utt-suffix -${lang_code} $sourcedir_babel/${lang_code}/data/train \
         data/tr_${lang_code}
        utils/copy_data_dir.sh --utt-suffix -${lang_code} $sourcedir_babel/${lang_code}/data/dev   \
         data/dt_${lang_code}
    done
    for lang_code in $target_babel_langs; do
        utils/copy_data_dir.sh --utt-suffix -${lang_code} $sourcedir_babel/${lang_code}/data/dev \
         data/dt_${lang_code}
        utils/copy_data_dir.sh --utt-suffix -${lang_code} $sourcedir_babel/${lang_code}/data/eval_${lang_code} \
         data/et_${lang_code}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}
fbankdir=fbank_babel

local/make_symlink_dir.sh --tmp-root $storage ${fbankdir}
if [ ${stage} -le 1 ]; then
    for i in ${train_babel_langs}; do
        if [ ! -f data/tr_$i/feats.scp ]; then
            x=tr_$i
	        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
            steps/make_fbank_pitch.sh --cmd "$train_cmd -js 5" --nj 50 data/${x} exp/make_fbank/${x} ${fbankdir}
            ./utils/fix_data_dir.sh data/${x} 
            compute-cmvn-stats scp:data/${x}/feats.scp data/${x}/cmvn.ark
        fi
        if [ ! -f data/dt_$i/feats.scp ]; then
            x=dt_$i
	        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
            steps/make_fbank_pitch.sh --cmd "$train_cmd -js 5" --nj 50 data/${x} exp/make_fbank/${x} ${fbankdir}
            ./utils/fix_data_dir.sh data/${x} 
            compute-cmvn-stats scp:data/${x}/feats.scp data/${x}/cmvn.ark
        fi
    done 
    utils/combine_data.sh data/${train_set}_org data/tr_101 data/tr_103 data/tr_104 data/tr_105 data/tr_404 data/tr_107 data/tr_201 data/tr_204 data/tr_205 data/tr_207
    utils/combine_data.sh data/${train_dev}_org data/dt_101 data/dt_103 data/dt_104 data/dt_105 data/dt_404 data/dt_107 data/dt_201 data/dt_204 data/dt_205 data/dt_207

    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}


    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/li10/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    elif [[ $(hostname -f) == *.fit.vutbr.cz ]] && [ ! -d ${feat_tr_dir} ]; then
	local/make_symlink_dir.sh --tmp-root $storage ${feat_tr_dir}
    else
	mkdir -p ${feat_tr_dir}
    fi

    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/li10/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    elif [[ $(hostname -f) == *.fit.vutbr.cz ]] && [ ! -d ${feat_dt_dir} ]; then
	local/make_symlink_dir.sh --tmp-root $storage ${feat_dt_dir}
    else
	 mkdir -p ${feat_dt_dir}
    fi

    dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 5 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
fi

lang=data/lang_1char
dict=$lang/train_units.txt
nlsyms=$lang/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    ./local/make_dct.sh \
 	--lang $lang    \
 	--data "data/${train_set} data/${train_dev}"

    ./local/make_json.sh \
 	--lang $lang                    \
 	--data_in data/${train_set} --data ${feat_tr_dir}

    ./local/make_json.sh \
	--lang $lang                    \
	--data_in data/${train_dev} --data ${feat_dt_dir}
fi

if [ -z ${tag} ]; then
#    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_ctc${ctctype}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_epoch${epochs}_ngpu$ngpu
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
    addtmpopt=''
    [ $backend != "pytorch" ] && addtmpopt=_${backend}
    expdir=${expdir}${addtmpopt}
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
    ${cuda_cmd} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
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
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
fi

exit 0



## Adaptation of the language independant (LI) trained model towards target languages
#           ***************** has to be implemented ******************


if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}

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
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
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
            &
        wait

        score_sclite.sh --nlsyms ${nlsyms} --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

