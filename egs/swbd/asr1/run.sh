#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
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

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
swbd1_dir=/export/corpora3/LDC/LDC97S62
eval2000_dir="/export/corpora2/LDC/LDC2002S09/hub5e_00 /export/corpora2/LDC/LDC2002T43"
rt03_dir=/export/corpora/LDC/LDC2007S10

# Byte Pair Encoding
use_bpe=true
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

train_set=train_nodup
train_dev=train_dev
recog_set="train_dev eval2000 rt03"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/swbd1_data_download.sh ${swbd1_dir}
    local/swbd1_prepare_dict.sh
    local/swbd1_data_prep.sh ${swbd1_dir}
    local/eval2000_data_prep.sh ${eval2000_dir}
    local/rt03_data_prep.sh ${rt03_dir}
    # upsample audio from 8k to 16k to make a recipe consistent with others
    for x in train eval2000 rt03; do
	sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" ${data_root}/${x}/wav.scp
    done
    # normalize eval2000 ant rt03 texts by
    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    for x in eval2000 rt03; do
        cp ${data_root}/${x}/text ${data_root}/${x}/text.org
        paste -d "" \
            <(cut -f 1 -d" " ${data_root}/${x}/text.org) \
            <(awk '{$1=""; print tolower($0)}' ${data_root}/${x}/text.org | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") \
            | sed -e 's/\s\+/ /g' > ${data_root}/${x}/text
        # rm ${data_root}/${x}/text.org
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
    for x in train eval2000 rt03; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            ${data_root}/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh ${data_root}/${x}
    done

    utils/subset_data_dir.sh --first ${data_root}/train 4000 ${data_root}/${train_dev} # 5hr 6min
    n=$(($(wc -l < ${data_root}/train/segments) - 4000))
    utils/subset_data_dir.sh --last ${data_root}/train ${n} ${data_root}/train_nodev
    utils/${data_root}/remove_dup_utts.sh 300 ${data_root}/train_nodev ${data_root}/${train_set} # 286hr

    # compute global CMVN
    compute-cmvn-stats scp:${data_root}/${train_set}/feats.scp ${data_root}/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-${data_root}/egs/swbd/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-${data_root}/egs/swbd/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        ${data_root}/${train_set}/feats.scp ${data_root}/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
        ${data_root}/${train_dev}/feats.scp ${data_root}/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
            ${data_root}/${rtask}/feats.scp ${data_root}/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

# path to store train/dev json
if ${use_bpe}; then
    train_json=${feat_tr_dir}/data_${bpemode}${nbpe}.json
    valid_json=${feat_dt_dir}/data_${bpemode}${nbpe}.json

    bpemodel=${data_root}/lang_${bpemode}${nbpe}/${train_set}_${bpemode}${nbpe}
    dict=${data_root}/lang_${bpemode}${nbpe}/${train_set}_${bpemode}${nbpe}_units.txt
else
    train_json=${feat_tr_dir}/data.json
    valid_json=${feat_dt_dir}/data.json

    dict=${data_root}/lang_1char/${train_set}_units.txt
    nlsyms=${data_root}/lang_1char/non_lang_syms.txt
fi

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    if ${use_bpe}; then
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
        echo "Stage 2 (BPE) finish!"
    else
        mkdir -p ${data_root}/lang_1char/

        echo "make a non-linguistic symbol list"
        cut -f 2- ${data_root}/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
        cat ${nlsyms}

        echo "make a dictionary"
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        text2token.py -s 1 -n 1 -l ${nlsyms} ${data_root}/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
        wc -l ${dict}

        echo "make json files"
        data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
            ${data_root}/${train_set} ${dict} > ${train_json}
        data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
            ${data_root}/${train_dev} ${dict} > ${valid_json}
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            data2json.sh --feat ${feat_recog_dir}/feats.scp \
                --nlsyms ${nlsyms} ${data_root}/${rtask} ${dict} > ${feat_recog_dir}/data.json
        done
        echo "Stage 2 (char) finish!"
    fi
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${use_bpe}; then
        expname=${expname}_BPE${nbpe}
    fi
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
        --valid-json ${valid_json}
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
        if ${use_bpe}; then
            recog_json_name=data_${bpemode}${nbpe}
        else
            recog_json_name=data
        fi
        splitjson.py --parts ${nj} ${feat_recog_dir}/${recog_json_name}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/${recog_json_name}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        if ${use_bpe}; then
            score_sclite.sh --wer true --bpe ${nbpe} --bpemodel ${bpemodel}.model ${expdir}/${decode_dir} ${dict}
        else
            score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        fi
        local/score_sclite.sh ${data_root}/eval2000 ${expdir}/${decode_dir}
        local/score_sclite.sh ${data_root}/rt03 ${expdir}/${decode_dir}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

