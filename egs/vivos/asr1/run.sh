#!/usr/bin/env bash

# Copyright 2019 National Institute of Informatics (Hieu-Thi Luong)
#  Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1           # start from -1 to download VIVOS corpus
nj=8
stop_stage=100
ngpu=1
debugmode=1
dumpdir=dump
N=0
verbose=0
resume=

# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml
lm_config=conf/lm.yaml

# rmmlm related
use_lm=false       # false means do not use lm
use_wordlm=true    # false means use character lm
lm_vocabsize=7184
lmtag=
lm_resume=

# decoding parameter
recog_model=model.loss.best

# transformer related
n_average=5
use_valbest_average=false

# data
datadir=./downloads
vivos_root=${datadir}/vivos
data_url=https://zenodo.org/api/files/a3a96378-5e63-4bf3-8fa6-fe2bebc871c7/vivos.tar.gz

# exp tag
tag=""

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

train_set="train_nodev"
train_dev="train_dev"
lm_test="test"
recog_set="train_dev test"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data download"

    mkdir -p ${datadir}
    local/download_and_untar.sh ${datadir} ${data_url}
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p data/{train,test} exp

    if [ ! -f ${vivos_root}/README ]; then
        echo "Cannot find vivos root! Exiting..."
        exit 1;
    fi

    for x in test train; do
        awk -v dir=${vivos_root}/$x '{ split($1,args,"_"); spk=args[1]; print $1" "dir"/waves/"spk"/"$1".wav" }' ${vivos_root}/$x/prompts.txt | sort > data/$x/wav.scp
        awk '{ split($1,args,"_"); spk=args[1]; print $1" "spk }' ${vivos_root}/$x/prompts.txt | sort > data/$x/utt2spk
        sort ${vivos_root}/$x/prompts.txt > data/$x/text
        utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Feature generation"

    fbankdir=fbank

    for x in test train; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
                                  data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/subset_data_dir.sh --first data/train 100 data/${train_dev}
    n=$(($(wc -l < data/train/text) - 100))
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
                data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
                ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Dictionary and Json Data Preparation"

    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict}
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    data2json.sh --feat ${feat_tr_dir}/feats.scp \
                 data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
                 data/${train_dev} ${dict} > ${feat_dt_dir}/data.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
                     data/${rtask} ${dict} > ${feat_recog_dir}/data.json
  done
fi

if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi

lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ ${use_lm} = true ]; then
    echo "Stage 3: LM Preparation"

    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}

        cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
        cut -f 2- -d" " data/${lm_test}/text > ${lmdatadir}/test.txt

        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}

        text2token.py -s 1 -n 1 data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 data/${lm_test}/text \
            | cut -f 2- -d" " > ${lmdatadir}/test.txt
    fi

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
                --test-label ${lmdatadir}/test.txt \
                --resume ${lm_resume} \
                --dict ${lmdict}
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
  echo "Stage 4: Network Training"

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


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Decoding"

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then

        if [ ${use_valbest_average} == true ]; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi

        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}
    fi

    pids=()
    for rtask in ${recog_set}; do
        (
            if [ ${use_lm} = true ]; then
                decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
                if [ ${use_wordlm} = true ]; then
                    recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
                else
                    recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
                fi
            else
                decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
                recog_opts=""
            fi

            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

            splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

            ngpu=0

            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                          asr_recog.py \
                          --config ${decode_config} \
                          --ngpu ${ngpu} \
                          --backend ${backend} \
                          --debugmode ${debugmode} \
                          --verbose ${verbose} \
                          --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
                          --result-label ${expdir}/${decode_dir}/data.JOB.json \
                          --model ${expdir}/results/${recog_model} \
                          ${recog_opts}

            score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
        ) &

        pids+=($!)
    done

    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi
