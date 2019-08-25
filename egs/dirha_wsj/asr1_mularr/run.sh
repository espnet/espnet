#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2019 Johns Hopkins University (Ruizhi Li)
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
seed=1

# feature configuration
do_delta=false

# config files
pretrain_preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
pretrain_train_config=conf/tuning/train_rnn.yaml
pretrain_decode_config=conf/tuning/decode_rnn.yaml
preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/tuning_mularr/train_mularr2_rnn.yaml
decode_config=conf/tuning_mularr/decode_mularr2_rnn.yaml
lm_config=conf/tuning/lm.yaml
pretrain_config=conf/tuning_mularr/pretrain_enc_att_ctc_dec.yaml # applicable when use_pretrain=True
freeze_pretrain_params=false
use_pretrain=false

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
n_average=10 # use 1 for RNN models
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
mics="Beam_Circular_Array Beam_Linear_Array" # must be to elements
mics_id=BCA_BLA #
dirha_wsj_folder=/export/b08/ruizhili/data/Data_processed
wsj1=/export/corpora5/LDC/LDC94S13B

# exp tag
pretrain_tag="" # pretrain tag for managing experiments.
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    # If YOU DON'T HAVE AUGMENTED DATA, FOLLOW THE FOLLOWING TWO STEPS BEFORE YOU RUN THE SCRIPT
    # FIRST: RUN THE FOLLOWING MATLAB FILE in ./local/tools/ TO PREPARE AUDIO FILES [chang your data directories accordingly]
    # Data_Contamination_dirha.m;
    # Data_Contamination_wsj0.m;
    # Data_Contamination_wsj1.m;

    # SECOND: RUN THE FOLLOWING TO COPY METADATA (prevous step doesnt copy meta data) [chang your data directories accordingly]
    # WSJ1_metadata was created using
    #     rsync -arv --exclude='*/*/*/*/*.wv1' --exclude='*/*/*/*/*.wv2' --exclude '*/*/*/*/*.wav' /export/b06/xwang/data/Data_processed/WSJ1_contaminated_mic_sum WSJ1_metadata
    # WSJ0_metadata was created using
    #     rsync -arv --exclude='*/*/*/*/*.wv1' --exclude='*/*/*/*/*.wv2' --exclude '*/*/*/*/*.wav' /export/b06/xwang/espnet_e2e/espnet/egs/dirha_wsj/Tools/LDC93S6B WSJ0_metadata
    # merge metadata with the created audio files
    #     rsync --recursive WSJ1_metadata/ WSJ1_contaminated_mic_Beam_Circular_Array/
    #     rsync --recursive WSJ0_metadata/ WSJ0_contaminated_mic_Beam_Circular_Array/

    for mic in $mics; do
        # augmented train
        wsj0_contaminated_folder=WSJ0_contaminated_mic_$mic # path of the wsj0 training data
	    wsj1_contaminated_folder=WSJ1_contaminated_mic_$mic # path of the wsj0 training data
        local/wsj_data_prep.sh ${dirha_wsj_folder}/$wsj0_contaminated_folder/??-{?,??}.? ${dirha_wsj_folder}/$wsj1_contaminated_folder/??-{?,??}.? || exit 1;
	    local/wsj_format_data.sh $mic || exit 1;

        # driha test
	    DIRHA_wsj_data=${dirha_wsj_folder}/DIRHA_wsj_oracle_VAD_mic_$mic # path of the test data
	    local/dirha_data_prep.sh $DIRHA_wsj_data/Sim dirha_sim_$mic  || exit 1;
	    local/dirha_data_prep.sh $DIRHA_wsj_data/Real dirha_real_$mic  || exit 1;
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    for mic in $mics; do

        train_set=train_si284_$mic
        train_dev=dirha_sim_$mic
        recog_set=dirha_real_$mic
        feat_tr_dir=${dumpdir}/${train_set}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
        feat_dt_dir=${dumpdir}/${train_set}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
        # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
        for x in ${train_set} ${train_dev} ${recog_set}; do
	        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
	            data/${x} exp/make_fbank/${x} ${fbankdir}
	        utils/fix_data_dir.sh data/${x}
        done
        # compute global CMVN
        compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

        # dump features for training
        if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
        utils/create_split_dir.pl \
            /export/b{10,11,12,13}/${USER}/espnet-data/egs/dirha_wsj/asr1_mularr/dump/${train_set}/${train_set}/delta${do_delta}/storage \
            ${feat_tr_dir}/storage
        fi
        if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
        utils/create_split_dir.pl \
            /export/b{10,11,12,13}/${USER}/espnet-data/egs/dirha_wsj/asr1_mularr/dump/${train_set}/${train_dev}/delta${do_delta}/storage \
            ${feat_dt_dir}/storage
        fi
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train_${train_set} ${feat_tr_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev_${train_set} ${feat_dt_dir}
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${train_set}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
            dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
                data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog_${train_set}/${rtask} \
                ${feat_recog_dir}
        done
    done
fi

dict=data/lang_1char/train_si284_Beam_Circular_Array_units.txt # pick one of the augmented channel [Beam_Circular_Array]
nlsyms=data/lang_1char/non_lang_syms.txt
lm_train=train_si284_Beam_Circular_Array # pick one of the augmented channel [Beam_Circular_Array]
lm_dev=dirha_sim_Beam_Circular_Array # pick one of the augmented channel [Beam_Circular_Array]
lm_test=dirha_real_Beam_Circular_Array # pick one of the augmented channel [Beam_Circular_Array]

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${lm_train}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${lm_train}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    for mic in $mics; do
        train_set=train_si284_$mic
        train_dev=dirha_sim_$mic
        recog_set=dirha_real_$mic
        feat_tr_dir=${dumpdir}/${train_set}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
        feat_dt_dir=${dumpdir}/${train_set}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
        data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
             data/${train_set} ${dict} > ${feat_tr_dir}/data.json
        data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
             data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
        for rtask in ${recog_set}; do
            feat_recog_dir=${dumpdir}/${train_set}/${rtask}/delta${do_delta}
            data2json.sh --feat ${feat_recog_dir}/feats.scp \
                --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
        done
    done
fi

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"

    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${lm_train}/text > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        cut -f 2- -d" " data/${lm_dev}/text > ${lmdatadir}/valid.txt
        cut -f 2- -d" " data/${lm_test}/text > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${lm_train}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${lm_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${lm_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi

    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
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

train_set=train_si284_${mics_id}_serial
train_dev=dirha_sim_${mics_id}_serial
recog_set= ;for mic in ${mics}; do recog_set+="dirha_real_${mic} "; done
feat_tr_dir=${dumpdir}/${train_set}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_set}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && [ ${use_pretrain} = true ]; then
    echo "stage 4: Prepare data for pre-train"

    # add suffix to the utt_names
    for mic in $mics; do
        # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
        for x in train_si284_$mic dirha_sim_$mic; do
            mkdir -p data/${x}_serial
            for file in feats.scp  text  utt2num_frames  utt2spk  wav.scp; do
                awk -v var="_$mic" '{ sub($1, $1var, $1); print }' data/${x}/${file} > data/${x}_serial/${file}
            done
            cp data/${x}/spk2gender data/${x}_serial/spk2gender
            utils/utt2spk_to_spk2utt.pl data/${x}_serial/utt2spk > data/${x}_serial/spk2utt
            fix_data_dir.sh data/${x}_serial || exit 1
        done
    done

    # create a list for multi-array training after pretrain
    train_mularr_list=
    for mic in $mics; do
        for x in train_si284_$mic dirha_sim_$mic; do
            train_mularr_list+="${x} "
        done
    done

    # combine train and dev
    # train
    datadirs=
    for mic in ${mics}; do datadirs+="data/train_si284_${mic}_serial ";done
    utils/combine_data.sh data/${train_set} $datadirs
    # dev
    datadirs=
    for mic in ${mics}; do datadirs+="data/dirha_sim_${mic}_serial ";done
    utils/combine_data.sh data/${train_dev} $datadirs

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/dirha_wsj/asr1_mularr/dump/${train_set}/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/dirha_wsj/asr1_mularr/dump/${train_set}/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train_${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev_${train_set} ${feat_dt_dir}
    for rtask in ${recog_set} ${train_mularr_list}; do
        feat_recog_dir=${dumpdir}/${train_set}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog_${train_set}/${rtask} \
            ${feat_recog_dir}
    done

    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set} ${train_mularr_list}; do
        feat_recog_dir=${dumpdir}/${train_set}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

if [ -z ${pretrain_tag} ]; then
    expname=${train_set}_${backend}_$(basename ${pretrain_train_config%.*})_$(basename ${pretrain_preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${pretrain_tag}
fi
expdir=exp_pretrain/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && [ ${use_pretrain} = true ]; then
    echo "stage 5: Network Training (pretrain)"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${pretrain_train_config} \
        --preprocess-conf ${pretrain_preprocess_config} \
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
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && [ ${use_pretrain} = true ]; then
    echo "stage 6: Decoding (pretrain)"
    nj=32
    if [[ $(get_yaml.py ${pretrain_train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${recog_model} \
                               --num ${n_average}
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${pretrain_decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/${train_set}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${pretrain_decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${use_pretrain} = true ]; then
    train_set=train_si284_${mics_id}_with_pretrain
    train_dev=dirha_sim_${mics_id}_with_pretrain
    recog_set=dirha_real_${mics_id}_with_pretrain
else
    train_set=train_si284_${mics_id}
    train_dev=dirha_sim_${mics_id}
    recog_set=dirha_real_${mics_id}
fi
feat_tr_dir=${dumpdir}/${train_set}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_set}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
feat_recog_dir=${dumpdir}/${train_set}/${recog_set}/delta${do_delta}; mkdir -p ${feat_recog_dir}

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Create paralell data.json"

    if [ ${use_pretrain} = true ]; then
        pretrain_train_data=train_si284_${mics_id}_serial

        featscps= # train
        for mic in $mics; do featscps+="${dumpdir}/${pretrain_train_data}/train_si284_$mic/delta${do_delta}/feats.scp,";done
        data2json.sh --feat ${featscps} \
            --nlsyms ${nlsyms} data/train_si284_Beam_Circular_Array ${dict} > ${feat_tr_dir}/data.json

        featscps= # dev
        for mic in $mics; do featscps+="${dumpdir}/${pretrain_train_data}/dirha_sim_$mic/delta${do_delta}/feats.scp,";done
        data2json.sh --feat ${featscps} \
            --nlsyms ${nlsyms} data/dirha_sim_Beam_Circular_Array ${dict} > ${feat_dt_dir}/data.json

        featscps= # test
        for mic in $mics; do featscps+="${dumpdir}/${pretrain_train_data}/dirha_real_$mic/delta${do_delta}/feats.scp,";done
        data2json.sh --feat ${featscps} \
            --nlsyms ${nlsyms} data/dirha_real_Beam_Circular_Array ${dict} > ${feat_recog_dir}/data.json
    else
        featscps= # train
        for mic in $mics; do featscps+="${dumpdir}/train_si284_$mic/train_si284_$mic/delta${do_delta}/feats.scp,";done
        data2json.sh --feat ${featscps} \
            --nlsyms ${nlsyms} data/train_si284_Beam_Circular_Array ${dict} > ${feat_tr_dir}/data.json

        featscps= # dev
        for mic in $mics; do featscps+="${dumpdir}/train_si284_$mic/dirha_sim_$mic/delta${do_delta}/feats.scp,";done
        data2json.sh --feat ${featscps} \
            --nlsyms ${nlsyms} data/dirha_sim_Beam_Circular_Array ${dict} > ${feat_dt_dir}/data.json

        featscps= # test
        for mic in $mics; do featscps+="${dumpdir}/train_si284_$mic/dirha_real_$mic/delta${do_delta}/feats.scp,";done
        data2json.sh --feat ${featscps} \
            --nlsyms ${nlsyms} data/dirha_real_Beam_Circular_Array ${dict} > ${feat_recog_dir}/data.json
    fi
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
if [ ${use_pretrain} = true ]; then
    trained_model="$expdir/results/model.acc.best"
    expdir=exp_mularr_with_pretrain/${expname}_$(basename ${pretrain_config%.*})_fix${freeze_pretrain_params}
    mkdir -p ${expdir}

strings=$(python -c "\
import io, yaml
with io.open(\"$pretrain_config\", encoding='utf-8') as f:
  pretrain_conf = yaml.safe_load(f)
strings = \"\"
for key in pretrain_conf.keys():
  strings += \" -a {}.trained_model=${trained_model} -a {}.freeze_params=${freeze_pretrain_params}\".format(key, key)
print(strings)"
)
    change_yaml.py $pretrain_config -o $expdir/pretrain_conf.yaml ${strings}
    train_opts="--pretrain-conf $expdir/pretrain_conf.yaml"
else
    expdir=exp_mularr/${expname}
    train_opts=""
fi
mkdir -p ${expdir}

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
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
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        ${train_opts}
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Decoding"
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
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/${train_set}/${rtask}/delta${do_delta}

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
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
