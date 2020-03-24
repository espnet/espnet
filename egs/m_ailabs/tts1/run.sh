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
nj=32        # numebr of parallel jobs
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

# silence part trimming related
do_trimming=true
trim_threshold=60 # (in decibels)
trim_win_length=1024
trim_shift_length=256
trim_min_silence=0.01

# config files
train_config=conf/train_pytorch_tacotron2.yaml # you can select from conf or conf/tuning.
                                               # now we support tacotron2, transformer, and fastspeech.
                                               # see more info in the header of each config.
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# pretrained model related
tts_train_config=
pretrained_decoder_path=
ept_train_config="conf/ept.v1.yaml"
ept_decode_config="conf/ae_decode.yaml"
ept_eval=false

# Decoder pretraining (w/ ASR encoder) related
pretrained_encoder="librispeech.transformer_large"

# objective evaluation related
asr_model="librispeech.transformer.ngpu4"
eval_tts=true                                  # true: evaluate tts model, false: evaluate autoencoded speech

# dataset configuration
db_root=downloads
lang=en_US  # en_UK, de_DE, es_ES, it_IT
spk=judy    # see local/data_prep.sh to check available speakers

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

org_set=${lang}_${spk}
train_set=${lang}_${spk}_train
dev_set=${lang}_${spk}_dev
eval_set=${lang}_${spk}_eval

if ${do_trimming}; then
    org_set=${org_set}_trim
    train_set=${train_set}_trim
    dev_set=${dev_set}_trim
    eval_set=${eval_set}_trim
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download.sh ${db_root} ${lang}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root} ${lang} ${spk} data/${org_set}
    utils/fix_data_dir.sh data/${org_set}
    utils/validate_data_dir.sh --no-feats data/${org_set}

    # copy for fbank+pitch feature
    utils/copy_data_dir.sh data/${org_set} data/${org_set}_pitch
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
feat_tr_pitch_dir=${dumpdir}/${train_set}_pitch; mkdir -p ${feat_tr_pitch_dir}
feat_dt_pitch_dir=${dumpdir}/${dev_set}_pitch; mkdir -p ${feat_dt_pitch_dir}
feat_ev_pitch_dir=${dumpdir}/${eval_set}_pitch; mkdir -p ${feat_ev_pitch_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    
    echo "stage 1: Feature Generation"
    # Trim silence parts at the begining and the end of audio
    if ${do_trimming}; then
        echo "Trimmimg utterances..."
        trim_silence.sh --cmd "${train_cmd}" \
            --fs ${fs} \
            --win_length ${trim_win_length} \
            --shift_length ${trim_shift_length} \
            --threshold ${trim_threshold} \
            --min_silence ${trim_min_silence} \
            data/${org_set} \
            exp/trim_silence/${org_set}
    fi
    
    # Generate the fbank features; by default 80-dimensional on each frame
    fbankdir=fbank
    echo "Generating fbanks features..."
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
        
    # Generate the fbank+pitch features; by default (80+3)-dimensional on each frame
    echo "Generating fbanks features with pitch..."
    steps/make_fbank_pitch.sh --cmd "${train_cmd}" --nj ${nj} \
        --write_utt2num_frames true \
        data/${org_set}_pitch \
        exp/make_fbank/${org_set}_pitch \
        ${fbankdir}
    utils/fix_data_dir.sh data/${org_set}_pitch

    # make a dev set
    utils/subset_data_dir.sh --last data/${org_set} 500 data/${org_set}_tmp
    utils/subset_data_dir.sh --last data/${org_set}_tmp 250 data/${eval_set}
    utils/subset_data_dir.sh --first data/${org_set}_tmp 250 data/${dev_set}
    n=$(( $(wc -l < data/${org_set}/wav.scp) - 500 ))
    utils/subset_data_dir.sh --first data/${org_set} ${n} data/${train_set}
    rm -rf data/${org_set}_tmp

    utils/subset_data_dir.sh --last data/${org_set}_pitch 500 data/${org_set}_pitch_tmp
    utils/subset_data_dir.sh --last data/${org_set}_pitch_tmp 250 data/${eval_set}_pitch
    utils/subset_data_dir.sh --first data/${org_set}_pitch_tmp 250 data/${dev_set}_pitch
    n=$(( $(wc -l < data/${org_set}_pitch/wav.scp) - 500 ))
    utils/subset_data_dir.sh --first data/${org_set}_pitch ${n} data/${train_set}_pitch
    rm -rf data/${org_set}_tmp

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    compute-cmvn-stats scp:data/${train_set}_pitch/feats.scp data/${train_set}_pitch/cmvn.ark
    # for ASR input features: normalize using the cmvn of pretrained ASR model(`pretrained_model2` is the ASR model)
    cmvn_asr=$(find ${db_root}/${pretrained_encoder} -name "cmvn.ark" | head -n 1)

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${dev_set} ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${eval_set} ${feat_ev_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}_pitch/feats.scp ${cmvn_asr} exp/dump_feats/${train_set}_pitch ${feat_tr_pitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}_pitch/feats.scp ${cmvn_asr} exp/dump_feats/${dev_set}_pitch ${feat_dt_pitch_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}_pitch/feats.scp ${cmvn_asr} exp/dump_feats/${eval_set}_pitch ${feat_ev_pitch_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
    data2json.sh --feat ${feat_tr_pitch_dir}/feats.scp \
         data/${train_set}_pitch ${dict} > ${feat_tr_pitch_dir}/data.json
    data2json.sh --feat ${feat_dt_pitch_dir}/feats.scp \
         data/${dev_set}_pitch ${dict} > ${feat_dt_pitch_dir}/data.json
    data2json.sh --feat ${feat_ev_pitch_dir}/feats.scp \
         data/${eval_set}_pitch ${dict} > ${feat_ev_pitch_dir}/data.json
    
    # make json for encoder pretraining, using 80-d input and 80-d output
    local/make_ae_json.py --input-json ${feat_tr_dir}/data.json \
        --output-json ${feat_tr_dir}/data.json -O ${feat_tr_dir}/ae_data.json
    local/make_ae_json.py --input-json ${feat_dt_dir}/data.json \
        --output-json ${feat_dt_dir}/data.json -O ${feat_dt_dir}/ae_data.json
    local/make_ae_json.py --input-json ${feat_ev_dir}/data.json \
        --output-json ${feat_ev_dir}/data.json -O ${feat_ev_dir}/ae_data.json
        
    # make json for decoder pretraining, using 83-d input and 80-d output
    local/make_ae_json.py --input-json ${feat_tr_pitch_dir}/data.json \
        --output-json ${feat_tr_dir}/data.json -O ${feat_tr_dir}/pitch_ae_data.json
    local/make_ae_json.py --input-json ${feat_dt_pitch_dir}/data.json \
        --output-json ${feat_dt_dir}/data.json -O ${feat_dt_dir}/pitch_ae_data.json
    local/make_ae_json.py --input-json ${feat_ev_pitch_dir}/data.json \
        --output-json ${feat_ev_dir}/data.json -O ${feat_ev_dir}/pitch_ae_data.json
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Text-to-speech model training"

    mkdir -p ${expdir}

    tr_json=${feat_tr_dir}/data.json
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
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
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
    echo "stage 6: Objective Evaluation for TTS"

    for name in ${dev_set} ${eval_set}; do
        local/ob_eval/evaluate_cer.sh --nj ${nj} \
            --do_delta false \
            --eval_gt ${eval_gt} \
            --eval_tts ${eval_tts} \
            --db_root ${db_root} \
            --backend pytorch \
            --wer ${wer} \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${name} \
            ${spk}
    done

    echo "Finished."
fi

##############################################################

# Encoder pretraining

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Encoder pretraining"

    # Suggested usage:
    # 1. If the exp dirrectory of the TTS model uses tag, then specify both --tag and --tts_train_config
    #    else, specify --train_config
    # 2. Specify --model and --n_average 0
    # 3. Specify --ept_train_config

    # check input arguments
    if [[ ! -z ${tag} ]] && [[ -z ${tts_train_config} ]]; then
        echo "Please specify --tts_train_config (pre-trained TTS model config)"
        exit 1
    elif [ -z ${train_config} ]; then
        echo "Please specify --train_config"
    fi

    tts_train_config=${train_config}

    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}

    pretrained_decoder_path=${expdir}/results/${model}
    train_config="$(change_yaml.py \
        -a dec-init="${pretrained_decoder_path}" \
        -d model-module \
        -d batch-bins \
        -d accum-grad \
        -d epochs \
        -o "conf/$(basename "${tts_train_config}" .yaml).ept.yaml" "${tts_train_config}")"

    tr_json=${feat_tr_dir}/ae_data.json
    dt_json=${feat_dt_dir}/ae_data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/ept_train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/ept_results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --srcspk ${spk} \
           --trgspk ${spk} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config} \
           --config2 ${ept_train_config}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Encoder pretraining: decoding, synthesis, evaluation"
    
    # Suggested usage:
    # 1. If the exp dirrectory of the TTS model uses tag, then specify both --tag and --tts_train_config
    #    else, specify --train_config
    # 2. Specify --model and --n_average 0
    # 3. Specify --ept_train_config

    if [ -z ${tag} ]; then
        echo "Please specify --tag"
        exit 1
    fi
    expdir=exp/${train_set}_${backend}_${tag}
    outdir=${expdir}/ept_outputs_${model}

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
                --model ${expdir}/ept_results/${model} \
                --config ${ept_decode_config}
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
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
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
    
    echo "Objective Evaluation"
    for name in ${dev_set} ${eval_set}; do
        local/ob_eval/evaluate.sh --nj ${nj} \
            --do_delta false \
            --db_root ${db_root} \
            --backend pytorch \
            --wer true \
            --api v2 \
            ${asr_model} \
            ${outdir} \
            ${name} \
            ${spk}
    done

    echo "Finished."
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Mel autoencoder: generate hdf5"

    # generate h5 for WaveNet vocoder
    for name in ${dev_set} ${eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

###############################################################

expdir=exp/${expname}
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: decoder pretraining using ASR encoder"

    # check input arguments
    if [ -z ${tag} ]; then
        echo "Please specify exp tag."
        exit 1
    fi
    if [ -z ${pretrained_model2} ]; then
        echo "Please specify pre-trained asr model."
        exit 1
    fi
    if [ -z ${pretrained_model_path} ]; then
        echo "Please specify pre-trained tts model."
        exit 1
    fi

    # here, pretrained_model is the TTS decoder, pretrained_model2 is the ASR encoder
    # add pretrained model info in config
    
    # pretrained_model_path=$(find ${download_dir}/${pretrained_model} -name "model*.best" | head -n 1)
    pretrained_model2_path=$(find ${db_root}/${pretrained_model2} -name "model*.best" | head -n 1)
    train_config="$(change_yaml.py \
        -a pretrained-model="${pretrained_model_path}" \
        -a load-partial-pretrained-model="${load_partial_pretrained_model}" \
        -a pretrained-model2="${pretrained_model2_path}" \
        -a load-partial-pretrained-model2="${load_partial_pretrained_model2}" \
        -o "conf/$(basename "${train_config}" .yaml).${pretrained_model2}.yaml" "${train_config}")"

    tr_json=${feat_tr_dir}/asr_ae_data.json
    dt_json=${feat_dt_dir}/asr_ae_data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/second_ae_train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/second_ae_results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --srcspk ${spk} \
           --trgspk ${spk} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "stage 13: Pretrained decoder: decoding, Synthesis, and hdf5 generation"

    if [ -z ${tag} ]; then
        echo 'Please specify experiment tag'
        exit 1
    fi
    
    expname=${ae_train_set}_${tag}
    expdir=exp/${expname}
    outdir=${expdir}/${srcspk}_outputs_${model}_$(basename ${decode_config%.*})
    echo $outdir
    
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/ae_results/${model} \
                --model-conf ${expdir}/ae_results/model.json \
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
    
    # Synthesis
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
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
    
    # generate h5 for neural vocoder
    for name in ${pair_dev_set} ${pair_eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done

fi
