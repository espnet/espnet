#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=10        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=24000      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# silence part trimming related
trim_threshold=25 # (in decibels)
trim_win_length=1024
trim_shift_length=256
trim_min_silence=0.01

trans_type=char  # char or phn

# config files
train_config=conf/train_pytorch_transformer+spkemb.dec_train.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
voc=PWG                                         # GL or PWG
voc_expdir=../vc1_task1/downloads/pwg_task1     # If use provided pretrained models, set to desired dir, ex. `downloads/pwg_task1`
                                                # If use manually trained models, set to `../voc1/exp/<expdir>`
voc_checkpoint=                                 # If not specified, automatically set to the latest checkpoint 
griffin_lim_iters=64                            # the number of iterations of Griffin-Lim

# normalization related
cmvn=
norm_name=

# pretrained model related
pretrained_model_dir=downloads  # If use provided pretrained models, set to desired dir, ex. `downloads`
                                # If use manually trained models, set to `../libritts`
pretrained_model=               # use full path
finetuned_model_name=           # Only set to `tts1_[trgspk]`

# objective evaluation related
outdir=                                        # in case not executed together with decoding & synthesis stage
eval_model=true                                # true: evaluate trained model, false: evaluate ground truth
mcd=true                                       # true: evaluate MCD
mcep_dim=24
shift_ms=5

# dataset configuration
db_root=../vc1_task1/downloads/official_v1.0_training
eval_db_root=../vc1_task1/downloads/official_v1.0_training    # Same as `db_root` in training
list_dir=../vc1_task1/local/lists
spk=TEF1 
lang=Man

task1_spks=( "TEF1" "TEF2" "TEM1" "TEM2" )
task2_spks=( "TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1" )

# vc configuration
srcspk=                                         # Ex. SEF1
trgspk=                                         # Ex. TEF1

# exp tag
tag=""  # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

org_set=${spk}
train_set=${spk}_train
dev_set=${spk}_dev

################################################

# Target speaker dependent decoder training

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained model download"
    echo "Please download the dataset following the README."

    if [ ! -d ${pretrained_model_dir}/${pretrained_model_name} ]; then
        echo "Downloading pretrained TTS model..."
        local/pretrained_model_download.sh ${pretrained_model_dir} ${pretrained_model_name}
    fi
    echo "Pretrained TTS model exists: ${pretrained_model_name}"
    
    if [ ! -d ${voc_expdir} ]; then
        echo "Downloading pretrained PWG model..."
        local/pretrained_model_download.sh ${pretrained_model_dir} pwg_task1
    fi
    echo "PWG model exists: ${voc_expdir}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    if [ ! -e ${db_root} ]; then
        echo "${db_root} not found."
        echo "cd ${db_root}; ./run.sh --stop_stage -1; cd -"
        exit 1;
    fi

    # check speaker
    if $(echo ${task1_spks[*]} | grep -q ${spk}); then
        local/data_prep_task1.sh ${db_root} data/${org_set} ${spk} ${trans_type}
    elif $(echo ${task2_spks[*]} | grep -q ${spk}); then
        local/data_prep_task2.sh ${db_root} data/${org_set} ${lang} ${spk} ${trans_type}
    else 
        echo "Specified speaker ${spk} is not available."
        exit 1
    fi

    utils/data/resample_data_dir.sh ${fs} data/${org_set} # Downsample to fs from 24k
    utils/fix_data_dir.sh data/${org_set}
    utils/validate_data_dir.sh --no-feats data/${org_set}
fi

# check --cmvn and --norm_name
if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
fi

feat_tr_dir=${dumpdir}/${train_set}_${norm_name}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}_${norm_name}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    # Trim silence parts at the begining and the end of audio
    mkdir -p exp/trim_silence/${org_set}/figs  # avoid error
    trim_silence.sh --cmd "${train_cmd}" \
        --fs ${fs} \
        --win_length ${trim_win_length} \
        --shift_length ${trim_shift_length} \
        --threshold ${trim_threshold} \
        --min_silence ${trim_min_silence} \
        data/${org_set} \
        exp/trim_silence/${org_set}

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    fbankdir=fbank
    make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        data/${org_set} \
        exp/make_fbank/${org_set} \
        ${fbankdir}

    # make train/dev set according to lists
    lang_char=$(echo ${spk} | head -c 2 | tail -c 1)
    sed -e "s/^/${spk}_/" ${list_dir}/${lang_char}_train_list.txt > data/${org_set}/${lang_char}_train_list.txt
    sed -e "s/^/${spk}_/" ${list_dir}/${lang_char}_dev_list.txt > data/${org_set}/${lang_char}_dev_list.txt
    utils/subset_data_dir.sh --utt-list data/${org_set}/${lang_char}_train_list.txt data/${org_set} data/${train_set}
    utils/subset_data_dir.sh --utt-list data/${org_set}/${lang_char}_dev_list.txt data/${org_set} data/${dev_set}

    # check if --cmvn if specified
    if [ -z ${cmvn} ]; then
        echo "Please specify --cmvn ."
        exit 1
    fi

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp ${cmvn} exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp ${cmvn} exp/dump_feats/${dev_set} ${feat_dt_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    # make dummy dict
    dict="data/dummy_dict/X.txt"
    if [ ! -e ${dict} ]; then
        mkdir -p ${dict%/*}
        echo "<unk> 1" > ${dict}
    fi
    echo "dictionary: ${dict}"

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type ${trans_type} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type ${trans_type} \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in ${train_set} ${dev_set}; do
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
        utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj 1 --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
    done

    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # Extract x-vector
    for name in ${train_set} ${dev_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 \
            ${nnet_dir} data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
    # Update json
    for name in ${train_set} ${dev_set}; do
        local/update_json.sh ${dumpdir}/${name}_${norm_name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
    
    # make ae pair json
    local/make_pair_json.py \
        --src-json ${feat_tr_dir}/data.json \
        --trg-json ${feat_tr_dir}/data.json \
        -O ${feat_tr_dir}/ae_data.json
    local/make_pair_json.py \
        --src-json ${feat_dt_dir}/data.json \
        --trg-json ${feat_dt_dir}/data.json \
        -O ${feat_dt_dir}/ae_data.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoder training"
    
    # Suggested usage:
    # 1. Specify --pretrained_model (eg. exp/<..>/results/snapshot.ep.xxx)
    # 2. Specify --n_average 0
    # 3. Specify --train_config (original config for TTS) 

    # check input arguments
    if [ -z ${train_config} ]; then
        echo "Please specify --train_config"
        exit 1
    fi
    if [ -z ${tag} ]; then
        echo "Please specify --tag"
        exit 1
    fi
 
    expname=${train_set}_${backend}_${tag}
    expdir=exp/${expname}
    mkdir -p ${expdir}
    
    train_config="$(change_yaml.py \
        -a enc-init="${pretrained_model}" \
        -a dec-init="${pretrained_model}" \
        -o "conf/$(basename "${train_config}" .yaml).${tag}.yaml" "${train_config}")"
    
    # copy x-vector into expdir
    # empty the scp file
    xvec_dir=exp/xvector_nnet_1a/xvectors_${train_set}
    cp ${xvec_dir}/spk_xvector.ark ${expdir}
    sed "s~${xvec_dir}/~~" ${xvec_dir}/spk_xvector.scp > ${expdir}/spk_xvector.scp

    tr_json=${feat_tr_dir}/ae_data.json
    dt_json=${feat_dt_dir}/ae_data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoder training: Decoding, Synthesis"
    
    # Suggested usage:
    # 1. Specify --tag
    # 2. Specify --model
    # 3. Specify --norm_name
    # 4. Specify --cmvn
    # 5. Specify --voc. If use `PWG`, also specify --voc_expdir and CUDA_VISIBLE_DEVICES=

    if [ -z ${tag} ]; then
        echo "Please specify --tag"
        exit 1
    fi
    expdir=exp/${train_set}_${backend}_${tag}
    outdir=${expdir}/ae_outputs_${model}
    
    pids=() # initialize pids
    for name in ${dev_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}_${norm_name}/ae_data.json ${outdir}/${name}/data.json
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    
    echo "Synthesis"

    [ -z "${voc_checkpoint}" ] && voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t | head -n 1)"

    pids=() # initialize pids
    for name in ${dev_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        # check if --cmvn if specified
        if [ -z ${cmvn} ]; then
            echo "Please specify --cmvn ."
            exit 1
        fi
        apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp

        # GL
        if [ ${voc} = "GL" ]; then
            echo "Using Griffin-Lim phase recovery."
            convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                --iters ${griffin_lim_iters} \
                ${outdir}_denorm/${name} \
                ${outdir}_denorm/${name}/log \
                ${outdir}_denorm/${name}/wav
        # PWG
        elif [ ${voc} = "PWG" ]; then
            echo "Using Parallel WaveGAN vocoder."

            # variable settings
            voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
            voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
            wav_dir=${outdir}_denorm/${name}/pwg_wav
            hdf5_norm_dir=${outdir}_denorm/${name}/hdf5_norm
            [ ! -e "${wav_dir}" ] && mkdir -p ${wav_dir}
            [ ! -e ${hdf5_norm_dir} ] && mkdir -p ${hdf5_norm_dir}

            # normalize and dump them
            echo "Normalizing..."
            ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
                parallel-wavegan-normalize \
                    --skip-wav-copy \
                    --config "${voc_conf}" \
                    --stats "${voc_stats}" \
                    --feats-scp "${outdir}_denorm/${name}/feats.scp" \
                    --dumpdir ${hdf5_norm_dir} \
                    --verbose "${verbose}"
            echo "successfully finished normalization."

            # decoding
            echo "Decoding start. See the progress via ${wav_dir}/decode.log."
            ${cuda_cmd} --gpu 1 "${wav_dir}/decode.log" \
                parallel-wavegan-decode \
                    --dumpdir ${hdf5_norm_dir} \
                    --checkpoint "${voc_checkpoint}" \
                    --outdir ${wav_dir} \
                    --verbose "${verbose}"
            echo "successfully finished decoding."
        else
            echo "Vocoder type not supported. Only GL and PWG are available."
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

################################################

# Conversion

# Specify:
# 1. srcspk
# 2. trgspk
# 3. tag

if [ -z ${srcspk} ]; then
    echo "Please specify --srcspk."
    exit 1
fi

src_train_set=${srcspk}_train
src_dev_set=${srcspk}_dev
src_feat_tr_dir=${dumpdir}/${src_train_set}_${norm_name}; mkdir -p ${src_feat_tr_dir}
src_feat_dt_dir=${dumpdir}/${src_dev_set}_${norm_name}; mkdir -p ${src_feat_dt_dir}

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: Data preperation for source speaker"

    # Usage:
    # 1. Specify --cmvn
   
    local/data_prep_task1.sh ${db_root} data/${srcspk} ${srcspk} ${trans_type}
    utils/fix_data_dir.sh data/${srcspk}
    utils/validate_data_dir.sh --no-feats data/${srcspk}
    
    echo "Feature Generation"
    fbankdir=fbank
       
    make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        data/${srcspk} \
        exp/make_fbank/${srcspk}_${norm_name} \
        ${fbankdir}
    
    # make train/dev set according to lists
    sed -e "s/^/${srcspk}_/" ${list_dir}/E_train_list.txt > data/${srcspk}/E_train_list.txt
    sed -e "s/^/${srcspk}_/" ${list_dir}/E_dev_list.txt > data/${srcspk}/E_dev_list.txt
    utils/subset_data_dir.sh --utt-list data/${srcspk}/E_train_list.txt data/${srcspk} data/${src_train_set}
    utils/subset_data_dir.sh --utt-list data/${srcspk}/E_dev_list.txt data/${srcspk} data/${src_dev_set}
        
    if [ -z ${cmvn} ]; then
        echo "Please specify --cmvn."
        exit 1
    fi
    
    # dump features
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_train_set}/feats.scp ${cmvn} exp/dump_feats/${src_train_set} ${src_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_dev_set}/feats.scp ${cmvn} exp/dump_feats/${src_dev_set} ${src_feat_dt_dir}
    
    echo "Dictionary and Json Data Preparation"
    
    # make dummy dict
    dict="data/dummy_dict/X.txt"
    if [ ! -e ${dict} ]; then
        mkdir -p ${dict%/*}
        echo "<unk> 1" > ${dict}
    fi
    
    # make json labels
    data2json.sh --feat ${src_feat_tr_dir}/feats.scp \
         data/${src_train_set} ${dict} > ${src_feat_tr_dir}/data.json
    data2json.sh --feat ${src_feat_dt_dir}/feats.scp \
         data/${src_dev_set} ${dict} > ${src_feat_dt_dir}/data.json
fi

if [ -z ${trgspk} ]; then
    echo "Please specify --trgspk."
    exit 1
fi
if [ -z ${tag} ]; then
    echo "Please specify --tag."
    exit 1
fi

trg_train_set=${trgspk}_train
trg_dev_set=${trgspk}_dev
trg_feat_tr_dir=${dumpdir}/${trg_train_set}_${norm_name}; mkdir -p ${trg_feat_tr_dir}
trg_feat_dt_dir=${dumpdir}/${trg_dev_set}_${norm_name}; mkdir -p ${trg_feat_dt_dir}

pair=${srcspk}_${trgspk}
pair_train_set=${pair}_train
pair_dev_set=${pair}_dev
pair_feat_tr_dir=${dumpdir}/${pair_train_set}_${norm_name}; mkdir -p ${pair_feat_tr_dir}
pair_feat_dt_dir=${dumpdir}/${pair_dev_set}_${norm_name}; mkdir -p ${pair_feat_dt_dir}

expdir=exp/${trg_train_set}_${backend}_${tag}
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: Make pair json"

    cp ${src_feat_tr_dir}/data.json ${pair_feat_tr_dir}
    cp ${src_feat_dt_dir}/data.json ${pair_feat_dt_dir}

    # use the avg x-vector in target speaker training set
    echo "Updating x vector..."
    x_vector_ark=${expdir}/$(awk -v spk=$spk '/spk/{print $NF}' ${expdir}/spk_xvector.scp)
    sed "s~ ${srcspk}~ $x_vector_ark~" data/${src_train_set}/utt2spk > ${pair_feat_tr_dir}/xvector.scp
    sed "s~ ${srcspk}~ $x_vector_ark~" data/${src_dev_set}/utt2spk > ${pair_feat_dt_dir}/xvector.scp
    local/update_json.sh ${pair_feat_tr_dir}/data.json ${pair_feat_tr_dir}/xvector.scp
    local/update_json.sh ${pair_feat_dt_dir}/data.json ${pair_feat_dt_dir}/xvector.scp
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "stage 13: Decoding"
    
    # Usage:
    # 1. Specify --model
    # 2. Specify --voc. If use `PWG`, also specify --voc_expdir and CUDA_VISIBLE_DEVICES=
    
    if [[ -z ${tag} ]]; then
        echo "Please specify `tag` ."
        exit 1
    fi
    outdir=${expdir}/outputs_${model}

    echo "Decoding"
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_train_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}_${norm_name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Synthesis..."

    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_train_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        
        apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp

        # GL
        if [ ${voc} = "GL" ]; then
            echo "Using Griffin-Lim phase recovery."
            convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                --iters ${griffin_lim_iters} \
                ${outdir}_denorm/${name} \
                ${outdir}_denorm/${name}/log \
                ${outdir}_denorm/${name}/wav
        # PWG
        elif [ ${voc} = "PWG" ]; then
            echo "Using Parallel WaveGAN vocoder."

            # variable settings
            [ -z "${voc_checkpoint}" ] && voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
            voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
            voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
            wav_dir=${outdir}_denorm/${name}/pwg_wav
            hdf5_norm_dir=${outdir}_denorm/${name}/hdf5_norm
            [ ! -e "${wav_dir}" ] && mkdir -p ${wav_dir}
            [ ! -e ${hdf5_norm_dir} ] && mkdir -p ${hdf5_norm_dir}

            # normalize and dump them
            echo "Normalizing..."
            ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
                parallel-wavegan-normalize \
                    --skip-wav-copy \
                    --config "${voc_conf}" \
                    --stats "${voc_stats}" \
                    --feats-scp "${outdir}_denorm/${name}/feats.scp" \
                    --dumpdir ${hdf5_norm_dir} \
                    --verbose "${verbose}"
            echo "successfully finished normalization."

            # decoding
            echo "Decoding start. See the progress via ${wav_dir}/decode.log."
            ${cuda_cmd} --gpu 1 "${wav_dir}/decode.log" \
                parallel-wavegan-decode \
                    --dumpdir ${hdf5_norm_dir} \
                    --checkpoint "${voc_checkpoint}" \
                    --outdir ${wav_dir} \
                    --verbose "${verbose}"

            # renaming
            rename -f "s/_gen//g" ${wav_dir}/*.wav

            echo "successfully finished decoding."
        else
            echo "Vocoder type not supported. Only GL and PWG are available."
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    echo "stage 14: Objective Evaluation: ASR"

    for set_type in "dev"; do
        local/ob_eval/evaluate.sh --nj ${nj} \
            --eval_model ${eval_model} \
            --db_root ${db_root} \
            --vocoder ${voc} \
            --mcep_dim ${mcep_dim} \
            --shift_ms ${shift_ms} \
            ${outdir} ${set_type} ${srcspk} ${trgspk}
    done
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ] && $mcd; then
    echo "stage 15: Objective Evaluation: MCD"

    minf0=$(awk '{print $1}' conf/${trgspk}.f0)
    maxf0=$(awk '{print $2}' conf/${trgspk}.f0)
    for set_name in ${pair_dev_set}; do
        mcd_file=${outdir}_denorm/${set_name}/mcd.log

        # Decide wavdir depending on vocoder
        if [ ! -z ${voc} ]; then
            # select vocoder type (GL, PWG)
            if [ ${voc} == "PWG" ]; then
                wavdir=${outdir}_denorm/${set_name}/pwg_wav
            elif [ ${voc} == "GL" ]; then
                wavdir=${outdir}_denorm/${set_name}/wav
            else
                echo "Vocoder type other than GL, PWG is not supported!"
                exit 1
            fi
        else
            echo "Please specify vocoder."
            exit 1
        fi

        ${decode_cmd} ${mcd_file} \
            local/ob_eval/mcd_calculate.py \
                --wavdir ${wavdir} \
                --gtwavdir ${db_root}/${trgspk} \
                --mcep_dim ${mcep_dim} \
                --shiftms ${shift_ms} \
                --f0min ${minf0} \
                --f0max ${maxf0}
        grep 'Mean' ${mcd_file}
    done
fi
