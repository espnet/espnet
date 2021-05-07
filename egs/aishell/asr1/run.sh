#!/bin/bash

set -x

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch

stage=-1        # start from 0 if you need to start from data preparation
stop_stage=-1

ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/tuning/train_pytorch_transformer_maskctc.yaml

lm_config=conf/lm.yaml
decode_config=conf/tuning/decode_pytorch_transformer_maskctc_online_iter0_bl32.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=             # tag for managing LMs

# ngram
ngramtag=
n_gram=4
use_stearming=true
# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.

# data
data=/scratch/groups/swatana4/aishell/
data_url=www.openslr.org/resources/33

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp
train_dev=dev
recog_set="dev test"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${data} ${data_url} data_aishell
    local/download_and_untar.sh ${data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
    # remove space in text
    for x in train dev test; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
        rm data/${x}/text.org
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank

    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        data/train exp/make_fbank/train ${fbankdir}
    utils/fix_data_dir.sh data/train
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
        data/dev exp/make_fbank/dev ${fbankdir}
    utils/fix_data_dir.sh data/dev
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
        data/test exp/make_fbank/test ${fbankdir}
    utils/fix_data_dir.sh data/test

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 6 --write_utt2num_frames true \
        data/${train_set} exp/make_fbank/${train_set} ${fbankdir}
    utils/fix_data_dir.sh data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    split_dir=$(echo $PWD | awk -F "/" '{print $NF "/" $(NF-1)}')
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/a{11,12,13,14}/${USER}/espnet-data/egs/${split_dir}/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/a{11,12,13,14}/${USER}/espnet-data/egs/${split_dir}/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 6 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
		 data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
		     data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

ngramexpname=train_ngram
ngramexpdir=exp/${ngramexpname}
if [ -z ${ngramtag} ]; then
    ngramtag=${n_gram}
fi
mkdir -p ${ngramexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
        > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " \
        > ${lmdatadir}/valid.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
    
    lmplz --discount_fallback -o ${n_gram} <${lmdatadir}/train.txt > ${ngramexpdir}/${n_gram}gram.arpa
    build_binary -s ${ngramexpdir}/${n_gram}gram.arpa ${ngramexpdir}/${n_gram}gram.bin
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
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
        asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if ${use_stearming}; then
    recog_set="dev_unsegmented" # test_unsegmented"
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"

    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
	   [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
	   [[ $(get_yaml.py ${train_config} model-module) = *maskctc* ]] || \
	   [[ $(get_yaml.py ${train_config} etype) = transformer ]] || \
	   [[ $(get_yaml.py ${train_config} dtype) = transformer ]]; then
	average_opts=
	if ${use_valbest_average}; then
	    recog_model=model.val${n_average}.avg.best
	    average_opts="--log ${expdir}/results/log"
	else
	    recog_model=model.last${n_average}.avg.best
	fi
	average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average} \
			       ${average_opts}
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  
#            --rnnlm ${lmexpdir}/rnnlm.model.best 
#            --ngram-model ${ngramexpdir}/${n_gram}gram.bin
#            --api v2

        score_sclite.sh ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

set -x
if [ ${stage} -le 200 ] && [ ${stop_stage} -ge 200 ]; then
    dir=data/
    recog_set="dev"
    for task in ${recog_set}; do
	task_new=${task}_unsegmented
	if [ -d "${dir}/${task_new}" ]; then
	    rm -r ${dir}/${task_new}
	fi
	mkdir -p ${dir}/${task_new}/wavs
	python local/conct_dev_wav.py data/${task}/wav.scp data/${task_new}/wavs/ \
	       data/${task_new}/wavs/gen_wav.sh data/${task_new}/wav.scp
	bash data/${task_new}/wavs/gen_wav.sh
	cat ${dir}/${task}/text | python ./local/conct_dev.py > ${dir}/${task_new}/text
	cat ${dir}/${task_new}/text | cut -f 1 | awk {'print $1"\t"$1'} > ${dir}/${task_new}/utt2spk
	cp ${dir}/${task_new}/utt2spk ${dir}/${task_new}/spk2utt
    done
    fbankdir=fbank
    for task in ${recog_set}; do
	task_new=${task}_unsegmented
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true\
				  data/${task_new} exp/make_fbank/${task_new} ${fbankdir}
	utils/fix_data_dir.sh data/${task_new}
    done

    for task in ${recog_set}; do
	task_new=${task}_unsegmented
	feat_recog_dir=${dumpdir}/${task_new}/delta${do_delta}; mkdir -p ${feat_recog_dir}
	dump.sh --cmd ${train_cmd} --nj 20 --do_delta ${do_delta} \
		data/${task_new}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${task_new} ${feat_recog_dir}
    done

    for task in ${recog_set}; do
	task_new=${task}_unsegmented
	feat_recog_dir=${dumpdir}/${task_new}/delta${do_delta}
	data2json.sh --feat ${feat_recog_dir}/feats.scp \
		     data/${task_new} ${dict} > ${feat_recog_dir}/data.json
    done
fi

if [ ${stage} -le 201 ] && [ ${stop_stage} -ge 201 ]; then
    dir=data/
    recog_set="test"
    for task in ${recog_set}; do
	task_new=${task}_unsegmented
	if [ -d "${dir}/${task_new}" ]; then
	    rm -r ${dir}/${task_new}
	fi
	mkdir -p ${dir}/${task_new}/wavs
	python local/conct_wav.py data/${task}/wav.scp data/${task_new}/wavs/ \
	       data/${task_new}/wavs/gen_wav.sh data/${task_new}/wav.scp
	bash data/${task_new}/wavs/gen_wav.sh
	cat ${dir}/${task}/text | python ./local/conct.py > ${dir}/${task_new}/text
	cat ${dir}/${task_new}/text | cut -f 1 | awk {'print $1"\t"$1'} > ${dir}/${task_new}/utt2spk
	cp ${dir}/${task_new}/utt2spk ${dir}/${task_new}/spk2utt
    done
    
    fbankdir=fbank
    for task in ${recog_set}; do
	task_new=${task}_unsegmented
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true\
				  data/${task_new} exp/make_fbank/${task_new} ${fbankdir}
	utils/fix_data_dir.sh data/${task_new}
    done

    for task in ${recog_set}; do
	task_new=${task}_unsegmented
	feat_recog_dir=${dumpdir}/${task_new}/delta${do_delta}; mkdir -p ${feat_recog_dir}
	dump.sh --cmd ${train_cmd} --nj 20 --do_delta ${do_delta} \
		data/${task_new}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${task_new} ${feat_recog_dir}
    done

    for task in ${recog_set}; do
	task_new=${task}_unsegmented
	feat_recog_dir=${dumpdir}/${task_new}/delta${do_delta}
	data2json.sh --feat ${feat_recog_dir}/feats.scp \
		     data/${task_new} ${dict} > ${feat_recog_dir}/data.json
    done
fi
