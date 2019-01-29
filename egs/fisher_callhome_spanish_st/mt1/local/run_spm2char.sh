#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network architecture
# encoder related
etype=blstm     # encoder architecture type
elayers=2
eunits=1024
eprojs=1024
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=2
dunits=1024
# attention related
atype=add
adim=1024

# regualrization option
samp_prob=0
lsm_type=unigram
lsm_weight=0.1
drop_enc=0.3
drop_dec=0.3
weight_decay=0

# minibatch related
batchsize=32
maxlen_in=100  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=100 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=20
patience=3

# decoding parameter
beam_size=20
penalty=0.2
maxlenratio=10.0
minlenratio=0.0
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# bpemode (unigram or bpe)
nbpe=1000
bpemode=unigram

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

train_set=train_sp.en
train_dev=dev_sp.en
recog_set="fisher_dev.en fisher_dev2.en fisher_test.en callhome_devtest.en callhome_evltest.en"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    if [ ! -d "../st1/data" ]; then
    	echo "run ../st1/run.sh first"
    	exit 1
    fi

    for lang in es en; do
        utils/copy_data_dir.sh ../st1/data/train_sp.${lang}         data/train_sp.${lang}
        utils/copy_data_dir.sh ../st1/data/dev_sp.${lang}           data/dev_sp.${lang}
        utils/copy_data_dir.sh ../st1/data/fisher_dev.${lang}       data/fisher_dev.${lang}
        utils/copy_data_dir.sh ../st1/data/fisher_dev2.${lang}      data/fisher_dev2.${lang}
        utils/copy_data_dir.sh ../st1/data/fisher_test.${lang}      data/fisher_test.${lang}
        utils/copy_data_dir.sh ../st1/data/callhome_devtest.${lang} data/callhome_devtest.${lang}
        utils/copy_data_dir.sh ../st1/data/callhome_evltest.${lang} data/callhome_evltest.${lang}
    done
    # multi references
    for no in 1 2 3; do
        cp ../st1/data/fisher_dev.en/text.${no} data/fisher_dev.en/text.${no}
        cp ../st1/data/fisher_dev2.en/text.${no} data/fisher_dev2.en/text.${no}
        cp ../st1/data/fisher_test.en/text.${no} data/fisher_test.en/text.${no}
    done

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/fisher_callhome_spanish_st/st1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/fisher_callhome_spanish_st/st1/dump/${train_dev}/delta${do_delta}/storage \
          ${feat_dt_dir}/storage
    fi
fi

dict_tgt=data/lang_1char/train_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
dict_src=data/lang_1spm/train_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_1spm/train_${bpemode}${nbpe}
echo "dictionary (tgt): ${dict_tgt}"
echo "dictionary (src): ${dict_src}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    mkdir -p data/lang_1spm/

    echo "make a non-linguistic symbol list for all languages"
    cat data/train_sp.*/text | grep sp1.0 | cut -f 2- -d " " | grep -o -P '&[^;]*;|@-@' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    # Share the same dictinary between source and target languages
    echo "<unk> 1" > ${dict_tgt} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/train_sp.*/text | grep sp1.0 | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d " " | tr " " "\n" \
      | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict_tgt}
    wc -l ${dict_tgt}

    # Share the same dictinary between source and target languages
    echo "<unk> 1" > ${dict_src} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=`cat ${dict_src} | wc -l`
    cat data/train_sp.*/text | grep sp1.0 | cut -f 2- -d " " | grep -v -e '^\s*$' > data/lang_1spm/input.txt
    spm_train --user_defined_symbols=`cat ${nlsyms} | tr "\n" ","` --input=data/lang_1spm/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict_src}
    wc -l ${dict_src}

    # make json labels
    local/data2json.sh --nlsyms ${nlsyms} \
        data/${train_set} ${dict_tgt} > ${feat_tr_dir}/data_${bpemode}${nbpe}2char.json
    local/data2json.sh --nlsyms ${nlsyms} \
        data/${train_dev} ${dict_tgt} > ${feat_dt_dir}/data_${bpemode}${nbpe}2char.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        local/data2json.sh --nlsyms ${nlsyms} \
            data/${rtask} ${dict_tgt} > ${feat_recog_dir}/data_${bpemode}${nbpe}2char.json
    done

    # update json (add source references)
    for x in ${train_set}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/$(echo ${x} | cut -f -1 -d ".").es
        local/update_json.sh --bpecode ${bpemodel}.model --filter_speed_perturbation true \
            ${feat_dir}/data_${bpemode}${nbpe}2char.json ${data_dir} ${dict_src}
    done
    for x in ${train_dev} ${recog_set}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/$(echo ${x} | cut -f -1 -d ".").es
        local/update_json.sh --bpecode ${bpemodel}.model \
            ${feat_dir}/data_${bpemode}${nbpe}2char.json ${data_dir} ${dict_src}
    done

    # Fisher has 4 references per utterance
    for rtask in fisher_dev.en fisher_dev2.en fisher_test.en; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        for no in 1 2 3; do
          local/data2json.sh --text data/${rtask}/text.${no} --nlsyms ${nlsyms} \
              data/${rtask} ${dict_tgt} > ${feat_recog_dir}/data_${bpemode}${nbpe}2char_${no}.json
        done
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}${adim}_${opt}_sampprob${samp_prob}_lsm${lsm_weight}_drop${drop_enc}${drop_dec}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_wd${weight_decay}_${bpemode}${nbpe}2char
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        mt_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict-src ${dict_src} \
        --dict-tgt ${dict_tgt} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}2char.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}2char.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --dropout-rate ${drop_enc} \
        --dropout-rate-decoder ${drop_dec} \
        --opt ${opt} \
        --epochs ${epochs} \
        --patience ${patience} \
        --weight-decay ${weight_decay}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}
        mkdir -p ${expdir}/${decode_dir}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}2char.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}2char.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            &
        wait

        # Fisher has 4 references per utterance
        if [ ${rtask} = "fisher_dev.en" ] || [ ${rtask} = "fisher_dev2.en" ] || [ ${rtask} = "fisher_test.en" ]; then
            for no in 1 2 3; do
              cp ${feat_recog_dir}/data_${bpemode}${nbpe}2char_${no}.json ${expdir}/${decode_dir}/data_ref${no}.json
            done
        fi

        local/score_bleu.sh --set ${rtask} ${expdir}/${decode_dir} ${dict_tgt} ${dict_src}

    ) &
    done
    wait
    echo "Finished"
fi
