#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=2        #
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

# general configuration
data_root=data

# dropout
dropout_rate=0

# label smoothing
lsm_type=unigram
lsm_weight=

# BPE
bpemode=bpe
nbpe=500

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

train_dev=train_dev
train_set=train_nodup
recog_set="train_dev eval2000 rt03"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

# train/dev json path
train_json=${feat_tr_dir}/data_${bpemode}${nbpe}.json
valid_json=${feat_dt_dir}/data_${bpemode}${nbpe}.json

bpemodel=${data_root}/lang_${bpemode}${nbpe}/${train_set}_${bpemode}${nbpe}
dict=${data_root}/lang_${bpemode}${nbpe}/${train_set}_${bpemode}${nbpe}_units.txt

##############################################################################
# Please make sure you have run stage 1 in run.sh before running this script #
##############################################################################

### Task dependent. You have to check non-linguistic symbols used in the corpus.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p ${data_root}/lang_${bpemode}${nbpe}/

    # map acronym such as p._h._d. to p h d for train_set& dev_set
    cp ${data_root}/${train_set}/text ${data_root}/${train_set}/text.bpe.backup
    cp ${data_root}/${train_dev}/text ${data_root}/${train_dev}/text.bpe.backup
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' ${data_root}/${train_set}/text
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' ${data_root}/${train_dev}/text

    echo "make a dictionary"
    cut -f 2- -d" " ${data_root}/${train_set}/text \
        > ${data_root}/lang_${bpemode}${nbpe}/input.txt

    # Please make sure sentencepiece is installed
    spm_train --input=${data_root}/lang_${bpemode}${nbpe}/input.txt \
            --model_prefix=${bpemodel} \
            --vocab_size=${nbpe} \
            --character_coverage=1.0 \
            --model_type=${bpemode} \
            --input_sentence_size=100000000 \
            --bos_id=-1 \
            --eos_id=-1 \
            --unk_id=0 \
            --user_defined_symbols=[laughter],[noise],[vocalized-noise]

    spm_encode --model=${bpemodel}.model --output_format=piece < ${data_root}/lang_${bpemode}${nbpe}/input.txt \
                        > ${data_root}/lang_${bpemode}${nbpe}/encode.txt

    echo "<unk> 1" > ${dict}
    cat ${data_root}/lang_${bpemode}${nbpe}/encode.txt  | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        ${data_root}/${train_set} ${dict} > ${train_json}
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        ${data_root}/${train_dev} ${dict} > ${valid_json}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --allow-one-column true --bpecode ${bpemodel}.model\
            ${data_root}/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
    echo "BPE finish!"
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_BPE${nbpe}
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
        --train-json ${train_json} \
        --valid-json ${valid_json} \
        ${dropout_rate:+ --dropout-rate $dropout_rate --dropout-rate-decoder $dropout_rate} \
        ${lsm_weight:+ --lsm-weight $lsm_weight --lsm-type $lsm_type}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

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

        score_sclite.sh --wer true --bpe ${nbpe} --bpemodel ${bpemodel}.model ${expdir}/${decode_dir} ${dict}
        local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
        local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

