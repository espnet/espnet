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

# silence part trimming related
trim_threshold=25 # (in decibels)
trim_win_length=1024
trim_shift_length=256
trim_min_silence=0.01

# char or phn
# In the case of phn, input transcription is convered to phoneem using https://github.com/Kyubyong/g2p.
trans_type="phn"

# config files
train_config=conf/train_pytorch_transformer.v1.single.finetune.yaml # you can select from conf or conf/tuning.
                                                                    # now we support tacotron2 and transformer for TTS.
                                                                    # see more info in the header of each config.
decode_config=conf/decode.yaml

# decoding related
model=
n_average=1           # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# pretrained model related
download_dir=downloads
pretrained_model=  # see model info in local/pretrained_model_download.sh

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
db_root=../vc/downloads

# dataset configuration
available_spks=(
    "SEF1" "SEF2" "SEM1" "SEM2" "TEF1" "TEF2" "TEM1" "TEM2" "TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1"
)
available_langs=(
    "Eng" "Ger" "Fin" "Man" 
)
spk=TMF1  # see local/data_prep.sh to check available speakers
lang=Man
trg_lang=Eng
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
pseudo_data_tag=

# VCC2020 baseline: cascade ASR + TTS
list_file=${db_root}/lists/eval_list.txt 
transciption_file=${db_root}/prompts/Eng_transcriptions.txt # optional, should not be available at test time
#transciption_file= # optional, should not be available at test time
tts_model_dir=

# objective evaluation related
asr_model="librispeech.transformer.ngpu4"
eval_tts_model=true                            # true: evaluate tts model, false: evaluate ground truth
wer=true                                       # true: evaluate CER & WER, false: evaluate only CER
vocoder=

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
pseudo_train_set=${spk}_pseudo_train_${trg_lang}
pseudo_dev_set=${spk}_pseudo_dev_${trg_lang}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    # local/data_download.sh ${download_dir} ${spk}
    # local/pretrained_model_download.sh ${download_dir} ${pretrained_model}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root} ${spk} data/${org_set} ${trans_type} ${lang}
    utils/data/resample_data_dir.sh ${fs} data/${org_set} # Downsample to fs from 24k
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

    lang_char=$(echo ${spk} | head -c 2 | tail -c 1)
    
    # Trim silence parts at the begining and the end of audio
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

    # make train/dev/test set
    utils/subset_data_dir.sh --utt-list ${db_root}/lists/eval_list.txt data/${org_set} data/${eval_set}
    utils/subset_data_dir.sh --utt-list ${db_root}/lists/${lang_char}_train_list.txt data/${org_set} data/${train_set}
    utils/subset_data_dir.sh --utt-list ${db_root}/lists/${lang_char}_dev_list.txt data/${org_set} data/${dev_set}

    # use pretrained model cmvn
    cmvn=$(find ${download_dir}/${pretrained_model} -name "cmvn.ark" | head -n 1)

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp ${cmvn} exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta false \
        data/${dev_set}/feats.scp ${cmvn} exp/dump_feats/${dev_set} ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj 25 --do_delta false \
        data/${eval_set}/feats.scp ${cmvn} exp/dump_feats/${eval_set} ${feat_ev_dir}
fi

dict=$(find ${download_dir}/${pretrained_model} -name "*_units.txt" | head -n 1)
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    nlsyms=$(find ${download_dir}/${pretrained_model} -name "*_non_lang_syms.txt" | head -n 1)

    # make json labels using pretrained model dict
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms "${nlsyms}" --trans_type ${trans_type} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms "${nlsyms}" --trans_type ${trans_type} \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --nlsyms "${nlsyms}" --trans_type ${trans_type} \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
    
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in ${train_set} ${dev_set} ${eval_set}; do
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
    for name in ${train_set} ${dev_set} ${eval_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 \
            ${nnet_dir} data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
    # Update json
    for name in ${train_set} ${dev_set} ${eval_set}; do
        local/update_json.sh ${dumpdir}/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"

    # add pretrained model info in config
    pretrained_model_path=$(find ${download_dir}/${pretrained_model} -name "snapshot*" | head -n 1)
    if [ -z "$pretrained_model_path" ]; then
        pretrained_model_path=$(find ${download_dir}/${pretrained_model} -name "model.loss*" | head -n 1)
    fi
    if [ -z "$pretrained_model_path" ]; then
        echo "Cannot find pretrained model"
        exit 1
    fi
    train_config="$(change_yaml.py -a pretrained-model="${pretrained_model_path}" \
        -o "conf/$(basename "${train_config}" .yaml).${pretrained_model}.yaml" "${train_config}")"

    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
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

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding, Synthesis, hdf5 generation"
    
    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
    
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
    
    echo "Synthesis"
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
    
    echo "Generate hdf5"
    # generate h5 for WaveNet vocoder
    for name in ${dev_set} ${eval_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Objective Evaluation"

    for name in ${eval_set}; do
        local/ob_eval/evaluate_cer.sh --nj ${nj} \
            --do_delta false \
            --eval_tts_model ${eval_tts_model} \
            --db_root ${db_root} \
            --backend pytorch \
            --vocoder ${vocoder} \
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
    echo "stage 8: Make data json files for pseudo train/dev set"
    
    feat_pseudo_tr_dir=${dumpdir}/${pseudo_train_set}; mkdir -p ${feat_pseudo_tr_dir}
    feat_pseudo_dt_dir=${dumpdir}/${pseudo_dev_set}; mkdir -p ${feat_pseudo_dt_dir}
    nnet_dir=exp/xvector_nnet_1a

    local/data_prep_pseudo.sh ${db_root} ${spk} data/${pseudo_train_set} ${trans_type} ${trg_lang} train
    local/data_prep_pseudo.sh ${db_root} ${spk} data/${pseudo_dev_set} ${trans_type} ${trg_lang} dev
    data2json.sh --feat data/${pseudo_train_set}/feats.scp --trans_type ${trans_type} \
         data/${pseudo_train_set} ${dict} > ${feat_pseudo_tr_dir}/data.json
    data2json.sh --feat data/${pseudo_dev_set}/feats.scp --trans_type ${trans_type} \
         data/${pseudo_dev_set} ${dict} > ${feat_pseudo_dt_dir}/data.json

    # use the avg x-vector in training set
    x_vector_ark=$(awk -v spk=$spk '/spk/{print $NF}' ${nnet_dir}/xvectors_${train_set}/spk_xvector.scp)
    trg_lang_char=$(echo ${trg_lang} | head -c 1)

    tr_list_path=${db_root}/lists/${trg_lang_char}_train_list.txt
    sed "s~$~ $x_vector_ark~" ${tr_list_path} > ${feat_pseudo_tr_dir}/xvector.scp
    local/update_json.sh ${feat_pseudo_tr_dir}/data.json ${feat_pseudo_tr_dir}/xvector.scp
    
    dt_list_path=${db_root}/lists/${trg_lang_char}_dev_list.txt
    sed "s~$~ $x_vector_ark~" ${dt_list_path} > ${feat_pseudo_dt_dir}/xvector.scp
    local/update_json.sh ${feat_pseudo_dt_dir}/data.json ${feat_pseudo_dt_dir}/xvector.scp
    
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Decoding, Synthesis, hdf5 generation"
    
    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
    
    pids=() # initialize pids
    for name in ${pseudo_train_set} ${pseudo_dev_set}; do
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
    for name in ${pseudo_train_set} ${pseudo_dev_set}; do
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
    
    echo "Generate hdf5"
    # generate h5 for WaveNet vocoder
    for name in ${pseudo_train_set} ${pseudo_dev_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done

    # add full path to denormalized 
    for name in ${pseudo_train_set} ${pseudo_dev_set}; do
        sed 's?[EM]100[0-9][0-9] ?&'`pwd`'/?' ${outdir}_denorm/${name}/feats.scp > ${outdir}_denorm/${name}/feats_full.scp
    done

fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Objective Evaluation for training set"
    
    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

    local/ob_eval/evaluate_cer.sh --nj ${nj} \
        --do_delta false \
        --eval_tts_model ${eval_tts_model} \
        --db_root ${db_root} \
        --backend pytorch \
        --wer ${wer} \
        --vocoder ${vocoder} \
        --api v2 \
        ${asr_model} \
        ${outdir} \
        ${pseudo_train_set}

    echo "Finished."
fi

################################################

# VCC2020 baseline: cascade ASR + TTS

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: Recognize the eval set"

    expdir=exp/${srcspk}_eval_asr
    [ ! -e ${expdir} ] && mkdir -p ${expdir}
    
    local/recognize.sh --nj ${nj} \
        --db_root ${db_root} \
        --backend pytorch \
        --wer ${wer} \
        --api v2 \
        exp/${asr_model}_asr \
        ${expdir} \
        ${db_root}/${srcspk} \
        ${srcspk} \
        ${list_file} \
        ${transciption_file}

fi
    
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: Decoding, Synthesis, hdf5 generation"
    
    if [ -z ${tts_model_dir} ]; then
        echo "Please specify tts_model_dir!"
        exit 1
    fi
    pairname=${srcspk}_${trgspk}_eval
    outdir=exp/${srcspk}_eval_asr/$(basename ${tts_model_dir})_${model}; mkdir -p ${outdir}
    tts_datadir=exp/${srcspk}_eval_asr/data_tts/${trgspk}; mkdir -p ${tts_datadir}
    text=${tts_datadir}/text
    dict=$(find ${download_dir}/${pretrained_model} -name "*_units.txt" | head -n 1)
    nlsyms=$(find ${download_dir}/${pretrained_model} -name "*_non_lang_syms.txt" | head -n 1)

    echo "Data preparation..."
    # clean text from asr result
    echo "Cleaning text..."
    local/clean_text_asr_result.py \
        exp/${srcspk}_eval_asr/result/hyp.wrd.trn \
        $trans_type en_US > ${text}
    sed -i "s~${srcspk}~${trgspk}~g" ${text}

    # data2json.sh needs utt2spk
    echo "Making data.json ..."
    cp exp/${srcspk}_eval_asr/data_asr/utt2spk ${tts_datadir}/utt2spk
    sed -i "s~${srcspk}~${trgspk}~g" ${tts_datadir}/utt2spk
    data2json.sh --nlsyms "${nlsyms}" --trans_type ${trans_type} \
         ${tts_datadir} ${dict} > ${tts_datadir}/data.json
    
    # use the avg x-vector in target speaker training set
    # NOTE: I am actually not sure if we can only have spkemb as only input in data.json
    # will we need a dummy input1?
    echo "Updating x vector..."
    nnet_dir=exp/xvector_nnet_1a
    trgspk_train_set=${trgspk}_train_no_dev
    x_vector_ark=$(awk -v spk=$spk '/spk/{print $NF}' ${nnet_dir}/xvectors_${trgspk_train_set}/spk_xvector.scp)
    sed "s~ ${trgspk}~ $x_vector_ark~" ${tts_datadir}/utt2spk > ${tts_datadir}/xvector.scp
    local/update_json.sh ${tts_datadir}/data.json ${tts_datadir}/xvector.scp

    echo "Decoding"
    [ ! -e ${outdir}/${pairname} ] && mkdir -p ${outdir}/${pairname}
    cp ${tts_datadir}/data.json ${outdir}/${pairname}/
    splitjson.py --parts ${nj} ${outdir}/${pairname}/data.json
    # decode in parallel
    ${train_cmd} JOB=1:${nj} ${outdir}/${pairname}/log/decode.JOB.log \
        tts_decode.py \
            --backend ${backend} \
            --ngpu 0 \
            --verbose ${verbose} \
            --out ${outdir}/${pairname}/feats.JOB \
            --json ${outdir}/${pairname}/split${nj}utt/data.JOB.json \
            --model ${tts_model_dir}/results/${model} \
            --config ${decode_config}
    # concatenate scp files
    for n in $(seq ${nj}); do
        cat "${outdir}/${pairname}/feats.$n.scp" || exit 1;
    done > ${outdir}/${pairname}/feats.scp
    
    echo "Synthesis"
    [ ! -e ${outdir}_denorm/${pairname} ] && mkdir -p ${outdir}_denorm/${pairname}
    # use pretrained model cmvn
    cmvn=$(find ${download_dir}/${pretrained_model} -name "cmvn.ark" | head -n 1)
    apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
        scp:${outdir}/${pairname}/feats.scp \
        ark,scp:${outdir}_denorm/${pairname}/feats.ark,${outdir}_denorm/${pairname}/feats.scp
    convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        --iters ${griffin_lim_iters} \
        ${outdir}_denorm/${pairname}/ \
        ${outdir}_denorm/${pairname}/log \
        ${outdir}_denorm/${pairname}/wav
    
    echo "Generate hdf5"
    # generate h5 for WaveNet vocoder
    feats2hdf5.py \
        --scp_file ${outdir}_denorm/${pairname}/feats.scp \
        --out_dir ${outdir}_denorm/${pairname}/hdf5/
    (find "$(cd ${outdir}_denorm/${pairname}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${pairname}/hdf5_feats.scp
    echo "generated hdf5"

fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "stage 13: Objective Evaluation"
    
    pairname=${srcspk}_${trgspk}_eval
    outdir=exp/${srcspk}_eval_asr/$(basename ${tts_model_dir})_${model}

    local/ob_eval/evaluate_cer.sh --nj ${nj} \
        --do_delta false \
        --eval_tts_model ${eval_tts_model} \
        --db_root ${db_root} \
        --backend pytorch \
        --wer ${wer} \
        --vocoder ${vocoder} \
        --api v2 \
        ${asr_model} \
        ${outdir} \
        ${pairname}

    echo "Finished."
fi

###############################################################

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    echo "stage 14: [Teacher-forcing mode] Decoding, Synthesis, hdf5 generation"

    nj=10
    
    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    outdir=${expdir}/GTA_outputs_${model}_$(basename ${decode_config%.*})
    
    pids=() # initialize pids
    for name in ${train_set} ${dev_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --teacher-forcing True \
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
    for name in ${train_set} ${dev_set}; do
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
    
    echo "Generate hdf5"
    # generate h5 for WaveNet vocoder
    for name in ${train_set} ${dev_set}; do
        feats2hdf5.py \
            --scp_file ${outdir}_denorm/${name}/feats.scp \
            --out_dir ${outdir}_denorm/${name}/hdf5/
        (find "$(cd ${outdir}_denorm/${name}/hdf5; pwd)" -name "*.h5" -print &) | head > ${outdir}_denorm/${name}/hdf5_feats.scp
        echo "generated hdf5 for ${name} set"
    done
fi


################################################

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    echo "stage 15: Training set decoding, synthesizing"

    if [ -z ${model} ]; then
        echo "Please specify model!"
        exit 1
    fi
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

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
