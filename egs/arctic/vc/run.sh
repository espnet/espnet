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
nj=16        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 0, get more log)
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
use_pitch=false # use pitch or not

# config files
train_config=
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
voc=PWG                         # GL or PWG
voc_expdir=                     # If use provided pretrained models, set to desired dir, ex. `downloads/<...>`
                                # If use manually trained models, set to `../voc1/exp/<expdir>`
voc_checkpoint=                 # If not specified, automatically set to the latest checkpoint 
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# normalization related
src_cmvn=
trg_cmvn=
norm_name=

# pretrained model related
pretrained_model=           # requires full path

pretrained_asr_model=librispeech.transformer_large
pretrained_dec_model=
enc_init_mods="encoder"
dec_init_mods="decoder,postnet,feat_out,prob_out"

# non parallel training related
encoder_model_path=
decoder_model_path=
decoder_model_json=
np_outdir=

# objective evaluation related
outdir=                                        # in case not executed together with decoding & synthesis stage
eval_model=true                                # true: evaluate trained model, false: evaluate ground truth
mcd=true                                       # true: evaluate MCD
mcep_dim=24
shift_ms=5

# dataset configuration
db_root=downloads
srcspk=clb  # see local/data_prep.sh to check available speakers
trgspk=slt
specified_train_json=
num_train_utts=-1
aespk=

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
ae_train_set=${aespk}_train
ae_dev_set=${aespk}_dev
ae_eval_set=${aespk}_eval

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${db_root} ${srcspk}
    local/data_download.sh ${db_root} ${trgspk}
    # local/pretrained_model_download.sh ${db_root} ${pretrained_model}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
   
    for spk_org_set in ${src_org_set} ${trg_org_set}; do
        local/data_prep.sh ${db_root} ${spk_org_set} data/${spk_org_set}
        utils/fix_data_dir.sh data/${spk_org_set}
        utils/validate_data_dir.sh --no-feats data/${spk_org_set}
        # copy for fbank+pitch feature
        if ${use_pitch}; then
            utils/copy_data_dir.sh data/${spk_org_set} data/${spk_org_set}_pitch 
        fi
    done
fi

# Usage:
# --norm_name is always needed. It is used to specify features.

# Add "pitch" postfix if `use_pitch`
if ${use_pitch}; then
    src_org_set=${srcspk}_pitch
    src_train_set=${srcspk}_pitch_train
    src_dev_set=${srcspk}_pitch_dev
    src_eval_set=${srcspk}_pitch_eval
    trg_org_set=${trgspk}_pitch
    trg_train_set=${trgspk}_pitch_train
    trg_dev_set=${trgspk}_pitch_dev
    trg_eval_set=${trgspk}_pitch_eval
    pair_train_set=${pair}_pitch_train
    pair_dev_set=${pair}_pitch_dev
    pair_eval_set=${pair}_pitch_eval
fi

# check `norm_name`
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
        spk_feat_tr_dir=${dumpdir}/${spk_train_set}_${norm_name}
        spk_feat_dt_dir=${dumpdir}/${spk_dev_set}_${norm_name}
        spk_feat_ev_dir=${dumpdir}/${spk_eval_set}_${norm_name}
       
        if ${use_pitch}; then
            # Generate the fbank+pitch features; by default 80-dimensional fbanks on each frame
            steps/make_fbank_pitch.sh --cmd "${train_cmd}" --nj ${nj} \
                --write_utt2num_frames true \
                data/${spk_org_set} \
                exp/make_fbank/${spk_org_set}_${norm_name} \
                ${fbankdir}
            utils/fix_data_dir.sh data/${spk_org_set}
        else
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
        fi

        # make train/dev/eval set
        utils/subset_data_dir.sh --last data/${spk_org_set} 200 data/${spk_org_set}_tmp
        utils/subset_data_dir.sh --last data/${spk_org_set}_tmp 100 data/${spk_eval_set}
        utils/subset_data_dir.sh --first data/${spk_org_set}_tmp 100 data/${spk_dev_set}
        n=$(( $(wc -l < data/${spk_org_set}/wav.scp) - 200 ))
        utils/subset_data_dir.sh --first data/${spk_org_set} ${n} data/${spk_train_set}
        rm -rf data/${spk_org_set}_tmp
    done
        
    # If cmvn not specified, calculate using training features.
    # else, use specified cmvn.
    if [ -z ${src_cmvn} ]; then
        compute-cmvn-stats scp:data/${src_train_set}/feats.scp data/${src_train_set}/cmvn.ark
        src_cmvn=data/${src_train_set}/cmvn.ark
    fi
    if [ -z ${trg_cmvn} ]; then
        compute-cmvn-stats scp:data/${trg_train_set}/feats.scp data/${trg_train_set}/cmvn.ark
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
    if [ -e ${dict} ]; then
        mkdir -p ${dict%/*}
        echo "<unk> 1" > ${dict}
    fi
    
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

    # always use fbank features as target, so rename to fbank dirname
    # if use_pitch=false, this should not cause a change
    trg_feat_tr_dir=${dumpdir}/${trgspk}_train_${norm_name}
    trg_feat_dt_dir=${dumpdir}/${trgspk}_dev_${norm_name}
    trg_feat_ev_dir=${dumpdir}/${trgspk}_eval_${norm_name}

    # make pair json
    if [ ${num_train_utts} -ge 0 ]; then
        local/make_pair_json.py \
            --src-json ${src_feat_tr_dir}/data.json \
            --trg-json ${trg_feat_tr_dir}/data.json \
            -O ${pair_tr_dir}/data_n${num_train_utts}.json \
            --num_utts ${num_train_utts}
    else
        local/make_pair_json.py \
            --src-json ${src_feat_tr_dir}/data.json \
            --trg-json ${trg_feat_tr_dir}/data.json \
            -O ${pair_tr_dir}/data.json
    fi
    local/make_pair_json.py \
        --src-json ${src_feat_dt_dir}/data.json \
        --trg-json ${trg_feat_dt_dir}/data.json \
        -O ${pair_dt_dir}/data.json
    local/make_pair_json.py \
        --src-json ${src_feat_ev_dir}/data.json \
        --trg-json ${trg_feat_ev_dir}/data.json \
        -O ${pair_ev_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: VC model training"

    # Usage:
    # 1. Always specify --tag and --train_config
    # 2. Specify COMPLETE PATH for --pretrained_model (<...>/<...>/snapshot.ep.xxx) if needed.
    #    If not, train from scratch.
    # 3. --specified_train_json if needed.
    #    If not, not specifiying --num_train_utts will result in using full set.

    # add pretrained model info in config
    if [ ! -z ${pretrained_model} ]; then
        train_config="$(change_yaml.py \
            -a enc-init="${pretrained_model}" \
            -a enc-init-mods="${enc_init_mods}" \
            -a dec-init="${pretrained_model}" \
            -a dec-init-mods="${dec_init_mods}" \
            -o "conf/$(basename "${train_config}" .yaml).${tag}.yaml" "${train_config}")"
    fi

    if [[ -z ${tag} || -z {train_config} ]]; then
        echo "Please specify both `tag` and `train_config` tag."
        exit 1
    fi
    expname=${srcspk}_${trgspk}_${backend}_${tag}
    expdir=exp/${expname}

    if [ -z ${specified_train_json} ]; then
        if [ ${num_train_utts} -ge 0 ]; then
            tr_json=${pair_tr_dir}/data_n${num_train_utts}.json
        else
            tr_json=${pair_tr_dir}/data.json
        fi
    else
        tr_json=${specified_train_json}
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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding and synthesis"

    # Usage:
    # 1. Always specify --tag and --model.
    # 2. Specify --voc. If use `PWG`, also specify --voc_expdir and CUDA_VISIBLE_DEVICES=
    
    if [[ -z ${tag} ]]; then
        echo "Please specify `tag` ."
        exit 1
    fi
    expname=${srcspk}_${trgspk}_${backend}_${tag}
    expdir=exp/${expname}
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

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
        
        # If cmvn not specified, calculate using training features.
        # else, use specified cmvn.
        if [ -z ${trg_cmvn} ]; then
            trg_cmvn=data/${trgspk}_train/cmvn.ark # NOTE: not `trg_train_set` to avoid use_pitch?
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


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Objective Evaluation"

    for name in ${pair_dev_set} ${pair_eval_set}; do
        local/ob_eval/evaluate.sh --nj ${nj} \
            --eval_model ${eval_model} \
            --db_root ${db_root} \
            --vocoder ${voc} \
            --mcep_dim ${mcep_dim} \
            --shift_ms ${shift_ms} \
            ${outdir} ${name} ${srcspk} ${trgspk}
    done
fi

# TODO: write this part (stil not written in 20200410)
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Objective Evaluation for GT"

    for set_type in "dev" "eval"; do
        local/ob_eval/evaluate.sh --nj ${nj} \
            --eval_model false \
            --db_root ${db_root} \
            --mcd ${mcd} \
            --vocoder ${voc} \
            ${outdir} \
            ${srcspk}_${trgspk}_${set_type} \
            ${trgspk}_${set_type} \
            ${trgspk}
    done
fi
