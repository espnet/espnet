#!/bin/bash

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
lm_resume=     # Resume the LM training from snapshot

# data
voxforge=downloads # original data directory to be stored
lang=it # de, en, es, fr, it, nl, pt, ru

# feature configuration
do_delta=false

# transducer related
rnnt_mode='rnnt' # define transducer mode. Can be either 'rnnt' or 'rnnt-att'

# transducer config
train_config=conf/tuning/transducer/train_transducer.yaml
decode_config=conf/tuning/transducer/decode_transducer.yaml

# finetuning related
use_transfer=false # use transfer learning
type_transfer='enc' # define type of transfer lr. Can be either 'enc', 'dec' or 'both'

# model average related (only works when transformer part(s) are detected)
n_average=5
use_valbest_average=false

# experiment tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# datasets
train_set=tr_${lang}
train_dev=dt_${lang}
recog_set="dt_${lang} et_${lang}"

if [ ${use_transfer} == true ]; then
    dec_type=$(get_yaml.py ${train_config} dtype)
    dec_is_transformer="$( [[ $dec_type == *transformer* ]]; echo $? )"

    if [[ $dec_is_transformer -eq 0 && $type_transfer != 'enc' ]]; then
       echo "Finetuning: decoder init. for transformer model is not supported yet."
       echo "Finetuning: Switching to 'enc' transfer mode."

       type_transfer='enc'
    fi

    finetuning_conf=$(dirname ${train_config})/finetuning.yaml
    
    local/prep_transducer_finetuning.sh ${train_config}         \
                                        ${type_transfer}        \
                                        ${rnnt_mode}            \
                                        ${backend}              \
                                        --output ${finetuning_conf}

    exp_config=$(grep -e 'exp-conf' ${finetuning_conf} | cut -d' ' -f2)
    enc_config=$(grep -e "enc-conf" ${finetuning_conf} | cut -d' ' -f2 || echo "none")
    dec_config=$(grep -e "dec-conf" ${finetuning_conf} | cut -d' ' -f2 || echo "none")
else
    exp_config=${train_config}
    sed -i -r "s/(^rnnt-mode:) ('[a-z-]*')/\1 '${rnnt_mode}'/g" ${exp_config}
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/getdata.sh ${lang} ${voxforge}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    selected=${voxforge}/${lang}/extracted
    # Initial normalization of the data
    local/voxforge_data_prep.sh ${selected} ${lang}
    local/voxforge_format_data.sh ${lang}
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
        data/all_${lang} exp/make_fbank/train_${lang} ${fbankdir}
    utils/fix_data_dir.sh data/all_${lang}

    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 200 characters or 0 characters
    remove_longshortdata.sh data/all_${lang} data/all_trim_${lang}

    # following split consider prompt duplication (but does not consider speaker overlap instead)
    local/split_tr_dt_et.sh data/all_trim_${lang} data/tr_${lang} data/dt_${lang} data/et_${lang}
    rm -r data/all_trim_${lang}

    # compute global CMVN
    compute-cmvn-stats scp:data/tr_${lang}/feats.scp data/tr_${lang}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
        data/tr_${lang}/feats.scp data/tr_${lang}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data/dt_${lang}/feats.scp data/tr_${lang}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/tr_${lang}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/tr_${lang}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --lang ${lang} --feat ${feat_tr_dir}/feats.scp \
         data/tr_${lang} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --lang ${lang} --feat ${feat_dt_dir}/feats.scp \
         data/dt_${lang} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # If transfer learning is used, pre-training steps will be performed such as:
    ## 'enc' will add CTC model pre-training,
    ## 'dec' will add either LM ('rnnt') or attention-model ('rnnt-att') pre-training,
    ## 'both' will add both 'enc' and 'dec' pre-training steps.

    training_run() {
        # Given a transducer, CTC or attention config file as input,
        # run one network training step
        expname=${train_set}_${backend}_$(basename ${1%.*})
        if ${do_delta}; then
            expname=${expname}_delta
        fi

        expdir=exp/${expname}

        [ -f $expdir/results/model.loss.best ] && \
            [[ $expname != *"train_transducer"* ]] && return 0
        mkdir -p ${expdir}

        ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
                    asr_train.py \
                    --config $1 \
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
    }

    if [ $use_transfer == true ]; then
        if [ $type_transfer == 'enc' ] || [ $type_transfer == 'both' ]; then
            echo "Finetuning: Training CTC model using $enc_config"
            training_run "$enc_config"
        fi

        if [[ $rnnt_mode == 'rnnt' && \
                    ($type_transfer == 'dec' || $type_transfer == 'both') ]]; then
            echo "Finetuning: Training LM using $dec_config"

            lmexpname=train_rnnlm_${backend}_$(basename ${dec_config%.*})
            lmexpdir=exp/${lmexpname}

            if [ ! -f $lmexpdir/rnnlm.model.best ]; then
                mkdir -p ${lmexpdir}

                lmdatadir=data/local/lm_train
                lmdict=${dict}
                mkdir -p ${lmdatadir}
                text2token.py -s 1 -n 1 data/${train_set}/text \
                    | cut -f 2- -d" " > ${lmdatadir}/train.txt
                text2token.py -s 1 -n 1 data/${train_dev}/text \
                    | cut -f 2- -d" " > ${lmdatadir}/valid.txt
                text2token.py -s 1 -n 1 data/et_${lang}/text \
                    | cut -f 2- -d" " > ${lmdatadir}/test.txt

                ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
                            lm_train.py \
                            --config ${dec_config} \
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
        elif [[ $rnnt_mode == 'rnnt-att' && \
                      ($type_transfer == 'dec' || $type_transfer == 'both') ]]; then
            echo "Finetuning: Training attention model using $dec_config"
            training_run "$dec_config"
        fi
    fi
    echo "Main network: Training transducer model using ${exp_config}"
    training_run "${exp_config}"
fi

main_expdir=exp/${train_set}_${backend}_$(basename ${exp_config%.*})
if ${do_delta}; then
    main_expdir=${main_expdir}_delta
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    nj=16

    if [[ $(get_yaml.py ${train_config} etype) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = *transformer* ]]; then
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${main_expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi

        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${main_expdir}/results/snapshot.ep.* \
            --out ${main_expdir}/results/${recog_model} \
            --num ${n_average}
    else
        recog_model=model.loss.best
        opt=""
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

        ${decode_cmd} JOB=1:${nj} ${main_expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${main_expdir}/${decode_dir}/data.JOB.json \
            --model ${main_expdir}/results/${recog_model}

        score_sclite.sh --wer true ${main_expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
