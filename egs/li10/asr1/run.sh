#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a task of 10 language-indepent ASR used in
# S. Watanabe et al, "Language independent end-to-end architecture for
# joint language identification and speech recognition," Proc. ASRU'17, pp. 265--269 (2017)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_li10
train_dev=dev_li10
recog_set="dt_de dt_en dt_es dt_fr dt_it dt_ja dt_nl dt_pt dt_ru dt_zh et_de et_en et_es et_fr et_it et_ja_1 et_ja_2 et_ja_3 et_nl et_pt et_ru et_zh"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # TODO
    # add a check whether the following data preparation is completed or not
    # HKUST Mandarin
    lang_code=zh
    if [ -e ../../hkust/asr1/data ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../hkust/asr1/data/train_nodup_sp data/tr_${lang_code}
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../hkust/asr1/data/train_dev data/dt_${lang_code}
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../hkust/asr1/data/dev data/et_${lang_code}
    else
	echo "no hkust data directory found"
	echo "cd ../../hkust/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi

    # CSJ Japanese
    lang_code=ja
    if [ -e ../../csj/asr1/data ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/train_nodup data/tr_${lang_code}
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/train_dev data/dt_${lang_code}
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval1 data/et_${lang_code}_1
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval2 data/et_${lang_code}_2
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval3 data/et_${lang_code}_3
	# 1) change wide to narrow chars
	# 2) lower to upper chars
	for x in data/*_"${lang_code}"*; do
            utils/copy_data_dir.sh ${x} ${x}_org
            < ${x}_org/text nkf -Z |\
		awk '{for(i=2;i<=NF;++i){$i = toupper($i)} print}' > ${x}/text
            rm -fr ${x}_org
	done
    else
	echo "no csj data directory found"
	echo "cd ../../csj/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi

    # WSJ English
    lang_code=en
    if [ -e ../../wsj/asr1/data ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../wsj/asr1/data/train_si284 data/tr_${lang_code}
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../wsj/asr1/data/test_dev93 data/dt_${lang_code}
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../wsj/asr1/data/test_eval92 data/et_${lang_code}
    else
	echo "no wsj data directory found"
	echo "cd ../../wsj/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi

    # Voxforge
    for lang_code in de es fr it nl pt ru; do
	if [ -e ../../voxforge/asr1/data/tr_${lang_code} ]; then
            utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../voxforge/asr1/data/tr_${lang_code} data/tr_${lang_code}
            utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../voxforge/asr1/data/dt_${lang_code} data/dt_${lang_code}
            utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../voxforge/asr1/data/et_${lang_code} data/et_${lang_code}
	else
	    echo "no voxforge ${lang_code} data directory found"
	    echo "cd ../../voxforge/asr1/; ./run.sh --stop_stage 2 --lang ${lang_code}; cd -"
	    exit 1;
	fi
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    utils/combine_data.sh data/${train_set} data/tr_*
    utils/combine_data.sh data/${train_dev} data/dt_*

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/li10/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/li10/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --cmd "$train_cmd" --nj 80 --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --cmd "$train_cmd" --nj 32 --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --cmd "$train_cmd" --nj 32 --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
	recog_model=model.last${n_average}.avg.best
	average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
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
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        score_sclite.sh --nlsyms ${nlsyms} --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
