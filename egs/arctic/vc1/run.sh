#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=16        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# config files
train_config=
decode_config=conf/decode.yaml

# decoding related
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage
model=                      # VC Model checkpoint for decoding. If not specified, automatically set to the latest checkpoint
voc=PWG                     # vocoder used (GL or PWG)
griffin_lim_iters=64        # The number of iterations of Griffin-Lim

# pretrained model related
pretrained_model=           # available pretrained models: m_ailabs.judy.vtn_tts_pt

# dataset configuration
db_root=downloads
srcspk=clb                  # available speakers: "slt" "clb" "bdl" "rms"
trgspk=slt
num_train_utts=-1           # -1: use all 932 utts
norm_name=                  # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization

# exp tag
tag=""  # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

pair=${srcspk}_${trgspk}
src_org_set=${srcspk}
src_train_set=${srcspk}_train
src_dev_set=${srcspk}_dev
src_eval_set=${srcspk}_eval
trg_org_set=${trgspk}
trg_train_set=${trgspk}_train
trg_dev_set=${trgspk}_dev
trg_eval_set=${trgspk}_eval
pair_train_set=${pair}_train
pair_dev_set=${pair}_dev
pair_eval_set=${pair}_eval

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${db_root} ${srcspk}
    local/data_download.sh ${db_root} ${trgspk}

    # download pretrained model for training
    if [ -n "${pretrained_model}" ]; then
        local/pretrained_model_download.sh ${db_root} ${pretrained_model}
    fi

    # download pretrained PWG
    if [ ${voc} == "PWG" ]; then
        local/pretrained_model_download.sh ${db_root} pwg_${trgspk}
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    for spk_org_set in ${src_org_set} ${trg_org_set}; do
        local/data_prep.sh ${db_root}/cmu_us_${spk_org_set}_arctic ${spk_org_set} data/${spk_org_set}
        utils/fix_data_dir.sh data/${spk_org_set}
        utils/validate_data_dir.sh --no-feats data/${spk_org_set}
    done
fi

if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
fi
src_feat_tr_dir=${dumpdir}/${src_train_set}_${norm_name}; mkdir -p ${src_feat_tr_dir}
src_feat_dt_dir=${dumpdir}/${src_dev_set}_${norm_name}; mkdir -p ${src_feat_dt_dir}
src_feat_ev_dir=${dumpdir}/${src_eval_set}_${norm_name}; mkdir -p ${src_feat_ev_dir}
trg_feat_tr_dir=${dumpdir}/${trg_train_set}_${norm_name}; mkdir -p ${trg_feat_tr_dir}
trg_feat_dt_dir=${dumpdir}/${trg_dev_set}_${norm_name}; mkdir -p ${trg_feat_dt_dir}
trg_feat_ev_dir=${dumpdir}/${trg_eval_set}_${norm_name}; mkdir -p ${trg_feat_ev_dir}
pair_tr_dir=${dumpdir}/${pair_train_set}_${norm_name}; mkdir -p ${pair_tr_dir}
pair_dt_dir=${dumpdir}/${pair_dev_set}_${norm_name}; mkdir -p ${pair_dt_dir}
pair_ev_dir=${dumpdir}/${pair_eval_set}_${norm_name}; mkdir -p ${pair_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    # Generate the fbank features; by default 80-dimensional on each frame
    fbankdir=fbank
    for spk_org_set in ${src_org_set} ${trg_org_set}; do
        echo "Generating fbanks features for ${spk_org_set}..."

        spk_train_set=${spk_org_set}_train
        spk_dev_set=${spk_org_set}_dev
        spk_eval_set=${spk_org_set}_eval

        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${spk_org_set} \
            exp/make_fbank/${spk_org_set}_${norm_name} \
            ${fbankdir}

        # make train/dev/eval set
        utils/subset_data_dir.sh --last data/${spk_org_set} 200 data/${spk_org_set}_tmp
        utils/subset_data_dir.sh --last data/${spk_org_set}_tmp 100 data/${spk_eval_set}
        utils/subset_data_dir.sh --first data/${spk_org_set}_tmp 100 data/${spk_dev_set}
        n=$(( $(wc -l < data/${spk_org_set}/wav.scp) - 200 ))
        utils/subset_data_dir.sh --first data/${spk_org_set} ${n} data/${spk_train_set}
        rm -rf data/${spk_org_set}_tmp
    done

    # If not using pretrained models statistics, calculate in a speaker-dependent way.
    if [ -n "${pretrained_model}" ]; then
        src_cmvn="$(find "${db_root}/${pretrained_model}" -name "cmvn.ark" -print0 | xargs -0 ls -t | head -n 1)"
        trg_cmvn="$(find "${db_root}/${pretrained_model}" -name "cmvn.ark" -print0 | xargs -0 ls -t | head -n 1)"
    else
        compute-cmvn-stats scp:data/${src_train_set}/feats.scp data/${src_train_set}/cmvn.ark
        compute-cmvn-stats scp:data/${trg_train_set}/feats.scp data/${trg_train_set}/cmvn.ark
        src_cmvn=data/${src_train_set}/cmvn.ark
        trg_cmvn=data/${trg_train_set}/cmvn.ark
    fi

    # dump features
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_train_set}/feats.scp ${src_cmvn} exp/dump_feats/${src_train_set}_${norm_name} ${src_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_dev_set}/feats.scp ${src_cmvn} exp/dump_feats/${src_dev_set}_${norm_name} ${src_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_eval_set}/feats.scp ${src_cmvn} exp/dump_feats/${src_eval_set}_${norm_name} ${src_feat_ev_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_train_set}/feats.scp ${trg_cmvn} exp/dump_feats/${trg_train_set}_${norm_name} ${trg_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_dev_set}/feats.scp ${trg_cmvn} exp/dump_feats/${trg_dev_set}_${norm_name} ${trg_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_eval_set}/feats.scp ${trg_cmvn} exp/dump_feats/${trg_eval_set}_${norm_name} ${trg_feat_ev_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    # make dummy dict
    dict="data/dummy_dict/X.txt"
    mkdir -p ${dict%/*}
    echo "<unk> 1" > ${dict}

    # make json labels
    data2json.sh --feat ${src_feat_tr_dir}/feats.scp \
         data/${src_train_set} ${dict} > ${src_feat_tr_dir}/data.json
    data2json.sh --feat ${src_feat_dt_dir}/feats.scp \
         data/${src_dev_set} ${dict} > ${src_feat_dt_dir}/data.json
    data2json.sh --feat ${src_feat_ev_dir}/feats.scp \
         data/${src_eval_set} ${dict} > ${src_feat_ev_dir}/data.json
    data2json.sh --feat ${trg_feat_tr_dir}/feats.scp \
         data/${trg_train_set} ${dict} > ${trg_feat_tr_dir}/data.json
    data2json.sh --feat ${trg_feat_dt_dir}/feats.scp \
         data/${trg_dev_set} ${dict} > ${trg_feat_dt_dir}/data.json
    data2json.sh --feat ${trg_feat_ev_dir}/feats.scp \
         data/${trg_eval_set} ${dict} > ${trg_feat_ev_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Pair Json Data Preparation"

    # make pair json
    if [ ${num_train_utts} -ge 0 ]; then
        make_pair_json.py \
            --src-json ${src_feat_tr_dir}/data.json \
            --trg-json ${trg_feat_tr_dir}/data.json \
            -O ${pair_tr_dir}/data_n${num_train_utts}.json \
            --num_utts ${num_train_utts}
    else
        make_pair_json.py \
            --src-json ${src_feat_tr_dir}/data.json \
            --trg-json ${trg_feat_tr_dir}/data.json \
            -O ${pair_tr_dir}/data.json
    fi
    make_pair_json.py \
        --src-json ${src_feat_dt_dir}/data.json \
        --trg-json ${trg_feat_dt_dir}/data.json \
        -O ${pair_dt_dir}/data.json
    make_pair_json.py \
        --src-json ${src_feat_ev_dir}/data.json \
        --trg-json ${trg_feat_ev_dir}/data.json \
        -O ${pair_ev_dir}/data.json
fi

if [[ -z ${train_config} ]]; then
    echo "Please specify --train_config."
    exit 1
fi

# If pretrained model specified, add pretrained model info in config
if [ -n "${pretrained_model}" ]; then
    pretrained_model_path=$(find ${db_root}/${pretrained_model} -name "snapshot*" | head -n 1)
    train_config="$(change_yaml.py \
        -a enc-init="${pretrained_model_path}" \
        -a dec-init="${pretrained_model_path}" \
        -o "conf/$(basename "${train_config}" .yaml).${tag}.yaml" "${train_config}")"
fi
if [ -z ${tag} ]; then
    expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
else
    expname=${srcspk}_${trgspk}_${backend}_${tag}
fi
expdir=exp/${expname}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: VC model training"

    mkdir -p ${expdir}
    if [ ${num_train_utts} -ge 0 ]; then
        tr_json=${pair_tr_dir}/data_n${num_train_utts}.json
    else
        tr_json=${pair_tr_dir}/data.json
    fi
    dt_json=${pair_dt_dir}/data.json

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

if [ -z "${model}" ]; then
    model="$(find "${expdir}" -name "snapshot*" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
    model=$(basename ${model})
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding and synthesis"

    echo "Decoding..."
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
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
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}

        # Normalization
        # If not using pretrained models statistics, use statistics of target speaker
        if [ -n "${pretrained_model}" ]; then
            trg_cmvn="$(find "${db_root}/${pretrained_model}" -name "cmvn.ark" -print0 | xargs -0 ls -t | head -n 1)"
        else
            trg_cmvn=data/${trg_train_set}/cmvn.ark
        fi
        apply-cmvn --norm-vars=true --reverse=true ${trg_cmvn} \
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

            # check existence
            voc_expdir=${db_root}/pwg_${trgspk}
            if [ ! -d ${voc_expdir} ]; then
                echo "${voc_expdir} does not exist. Please download the pretrained model."
                exit 1
            fi

            # variable settings
            voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
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


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Objective Evaluation"

    for name in ${pair_dev_set} ${pair_eval_set}; do
        local/ob_eval/evaluate.sh --nj ${nj} \
            --db_root ${db_root} \
            --vocoder ${voc} \
            ${outdir} ${name}
    done
fi
