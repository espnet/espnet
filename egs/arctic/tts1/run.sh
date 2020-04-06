#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
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

# config files
train_config=conf/train_pytorch_transformer.v1.single.finetune.yaml # you can select from conf or conf/tuning.
                                                                    # now we support tacotron2 and transformer for TTS.
                                                                    # see more info in the header of each config.
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1           # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# pretrained model related
download_dir=downloads
pretrained_model="mailabs.en_US.judy.transformer.v1.single.R2"  # see model info in local/pretrained_model_download.sh

tts_train_config=
pretrained_model_path_ae=
load_partial_pretrained_model="encoder"
params_to_train="encoder"
ae_train_config="conf/ae_R2R2.yaml"

# objective evaluation related
asr_model="librispeech.transformer.ngpu4"
eval_tts_model=true                            # true: evaluate tts model, false: evaluate ground truth
wer=true                                       # true: evaluate CER & WER, false: evaluate only CER

# root directory of db
db_root=downloads

# dataset configuration
spk=slt  # see local/data_prep.sh to check available speakers
train_idx_start=-1
train_idx_end=-1

# pseudo parallel data for nonparallel training
srcspk=
trgspk=
src_train_idx_start=
src_train_idx_end=
trg_train_idx_start=
trg_train_idx_end=
src_decoded_feat=
trg_decoded_feat=
vc_dump_dir=

# exp tag
tag=""  # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

org_set=${spk}
train_set=${spk}_train_no_dev
dev_set=${spk}_dev
eval_set=${spk}_eval

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${download_dir} ${spk}
    #local/pretrained_model_download.sh ${download_dir} ${pretrained_model}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep.sh ${download_dir}/cmu_us_${spk}_arctic ${spk} data/${org_set}
    utils/fix_data_dir.sh data/${org_set}
    utils/validate_data_dir.sh --no-feats data/${org_set}
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
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

    # make a dev set
    utils/subset_data_dir.sh --last data/${org_set} 200 data/${org_set}_tmp
    utils/subset_data_dir.sh --last data/${org_set}_tmp 100 data/${eval_set}
    utils/subset_data_dir.sh --first data/${org_set}_tmp 100 data/${dev_set}
    n=$(( $(wc -l < data/${org_set}/wav.scp) - 200 ))
    utils/subset_data_dir.sh --first data/${org_set} ${n} data/${train_set}
    rm -rf data/${org_set}_tmp

    # use pretrained model cmvn
    cmvn=$(find ${download_dir}/${pretrained_model} -name "cmvn.ark" | head -n 1)

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp ${cmvn} exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp ${cmvn} exp/dump_feats/${dev_set} ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp ${cmvn} exp/dump_feats/${eval_set} ${feat_ev_dir}
fi

dict=$(find ${download_dir}/${pretrained_model} -name "*_units.txt" | head -n 1)
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    # make json labels using pretrained model dict
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
    
    if [ ${train_idx_start} -ge 0 ]; then
        local/make_partial_json.py --json_file ${feat_tr_dir}/data.json \
             -O ${feat_tr_dir}/data_${train_idx_start}_${train_idx_end}.json \
             --start ${train_idx_start} --end ${train_idx_end}
        local/make_partial_json.py --json_file ${feat_dt_dir}/data.json \
             -O ${feat_dt_dir}/data_${train_idx_start}_${train_idx_end}.json \
             --start ${train_idx_start} --end ${train_idx_end}
        local/make_partial_json.py --json_file ${feat_ev_dir}/data.json \
             -O ${feat_ev_dir}/data_${train_idx_start}_${train_idx_end}.json \
             --start ${train_idx_start} --end ${train_idx_end}
     fi
fi

# add pretrained model info in config
pretrained_model_path=$(find ${download_dir}/${pretrained_model} -name "model*.best" | head -n 1)
train_config="$(change_yaml.py -a pretrained-model="${pretrained_model_path}" \
    -o "conf/$(basename "${train_config}" .yaml).${pretrained_model}.yaml" "${train_config}")"

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Text-to-speech model training"

    if [ ${train_idx_start} -ge 0 ]; then
        tr_json=${feat_tr_dir}/data_${train_idx_start}_${train_idx_end}.json
    else
        tr_json=${feat_tr_dir}/data.json
    fi
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
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

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
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
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis"
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        # use pretrained model cmvn
        cmvn=$(find ${download_dir}/${pretrained_model} -name "cmvn.ark" | head -n 1)
        apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
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
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: generate hdf5"

    # generate h5 for WaveNet vocoder
    for name in ${dev_set} ${eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Objective Evaluation"

    for name in ${dev_set} ${eval_set}; do
        local/ob_eval/evaluate_cer.sh --nj ${nj} \
            --do_delta false \
            --eval_tts_model ${eval_tts_model} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${name}
    done

    echo "Finished."
fi

##################################################################

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Make data json files for autoencoder training"
    
    if [ ${train_idx_start} -ge 0 ]; then
        json_name=data_${train_idx_start}_${train_idx_end}.json
    else
        json_name=data.json
    fi

    # make pair json
    if [ ! -e ${feat_tr_dir}/ae_${json_name} ]; then
        echo "Making training json file"
        local/make_ae_json.py --json ${feat_tr_dir}/${json_name} -O ${feat_tr_dir}/ae_${json_name}
    fi
    if [ ! -e ${feat_dt_dir}/ae_${json_name} ]; then
        echo "Making development json file"
        local/make_ae_json.py --json ${feat_dt_dir}/${json_name} -O ${feat_dt_dir}/ae_${json_name}
    fi
    if [ ! -e ${feat_ev_dir}/ae_${json_name} ]; then
        echo "Making evaluation json file"
        local/make_ae_json.py --json ${feat_ev_dir}/${json_name} -O ${feat_ev_dir}/ae_${json_name}
    fi
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: encoder pretraining"
    
    if [ -z ${pretrained_model_path} ]; then
        echo "Please specify pre-trained tts model path."
        exit 1
    fi
    if [ -z ${tts_train_config} ]; then
        echo "Please specify pre-trained tts model config."
        exit 1
    fi
    
    train_config="$(change_yaml.py -a pretrained-model="${pretrained_model_path_ae}" \
        -a load-partial-pretrained-model="${load_partial_pretrained_model}" \
        -a params-to-train=encoder \
        -d model-module \
        -o "conf/$(basename "${tts_train_config}" .yaml).ae.yaml" "${tts_train_config}")"

    if [ ${train_idx_start} -ge 0 ]; then
        json_name=data_${train_idx_start}_${train_idx_end}.json
    else
        json_name=data.json
    fi
    tr_json=${feat_tr_dir}/ae_${json_name}
    dt_json=${feat_dt_dir}/ae_${json_name}
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/ae_train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/ae_results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --srcspk ${spk} \
           --trgspk ${spk} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config} \
           --config2 ${ae_train_config}
fi

outdir=${expdir}/ae_outputs_${model}
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Autoencoder decoding, synthesizing and hdf5 generation"

    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/ae_data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/ae_data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/ae_data.JOB.json \
                --model ${expdir}/ae_results/${model} \
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

    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        cmvn=$(find ${download_dir}/${pretrained_model} -name "cmvn.ark" | head -n 1) # TODO: should we use the cmvn in  data/${org_set} ?
        apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
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

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    # generate h5 for WaveNet vocoder
    for name in ${dev_set} ${eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

################################################3


if [ -z ${model} ]; then
    echo "Please specify model!"
    exit 1
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    echo "stage 15: Training set decoding, synthesizing"

    echo "Decoding"
    pids=() # initialize pids
    for name in ${train_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
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
    pids=() # initialize pids
    for name in ${train_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        # use pretrained model cmvn
        cmvn=$(find ${download_dir}/${pretrained_model} -name "cmvn.ark" | head -n 1)
        apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
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
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    echo "stage 16: Make data json files for pseudo data nonparallel training"
    
    if [ -z ${srcspk} ] || [ -z ${trgspk} ] \
        || [ -z ${src_train_idx_start} ] || [ -z ${src_train_idx_end} ] \
        || [ -z ${trg_train_idx_start} ] || [ -z ${trg_train_idx_end} ] \
        || [ -z ${src_decoded_feat} ] || [ -z ${trg_decoded_feat} ] \
        || [ -z ${vc_dump_dir} ]  ; then
        echo "Please specify needed arguments."
        exit 1
    fi

    local/make_pair_json.py \
        --src_json ${dumpdir}/${srcspk}_train_no_dev/data_${src_train_idx_start}_${src_train_idx_end}.json \
        --trg_json ${dumpdir}/${trgspk}_train_no_dev/data_${trg_train_idx_start}_${trg_train_idx_end}.json \
        --src_train_idx_start ${src_train_idx_start} \
        --trg_train_idx_start ${trg_train_idx_start} \
        --src_train_idx_end ${src_train_idx_end} \
        --trg_train_idx_end ${trg_train_idx_end} \
        --src_decoded_feat ${src_decoded_feat} \
        --trg_decoded_feat ${trg_decoded_feat} \
        --pwd $(pwd) \
        -O ${vc_dump_dir}/data_src_${src_train_idx_start}_${src_train_idx_end}_trg_${trg_train_idx_start}_${trg_train_idx_end}.json
fi
