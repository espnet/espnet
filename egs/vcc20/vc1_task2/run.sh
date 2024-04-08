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
nj=10        # numebr of parallel jobs
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

trans_type=  # char or phn

# config files
train_config=conf/train_pytorch_transformer+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
voc=PWG                         # GL or PWG
voc_expdir=downloads/pwg_task2  # If use provided pretrained models, set to desired dir, ex. `downloads/pwg_task2`
                                # If use manually trained models, set to `../voc1/exp/<expdir>`
voc_checkpoint=                 # If not specified, automatically set to the latest checkpoint
griffin_lim_iters=64            # the number of iterations of Griffin-Lim

# pretrained model related
pretrained_model_dir=downloads  # If use provided pretrained models, set to desired dir, ex. `downloads`
                                # If use manually trained models, set to `..`
pretrained_model_name=          # If use provided pretrained models, only set to `tts1_en_[de,fi,zh]`
                                # If use manually trained models, only set to `tts1_en_[de,fi,zh]`
finetuned_model_name=           # Only set to `tts1_en_[de,fi,zh]_[trgspk]`

# dataset configuration
db_root=../vc1_task1/downloads/vcc20
list_dir=local/lists
spk=TMF1
lang=Man

# vc configuration
srcspk=                                         # Ex. SEF1
trgspk=                                         # Ex. TMF1
asr_model="librispeech.transformer.ngpu4"
test_list_file=local/lists/eval_list.txt
test_name=eval_asr
tts_model_dir=                                  # If use downloaded model,
                                                # set to, ex. `downloads/tts1_en_zh_TMF1/exp/TMF1_train_pytorch_train_pytorch_transformer+spkemb.tts1_en_zh`
                                                # If use manually trained model,
                                                # set to, ex. `exp/TMF1_train_pytorch_train_pytorch_transformer+spkemb.tts1_en_zh`

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

# TTS training (finetuning)

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained model download"
    local/data_download.sh ${db_root}

    if [ ! -d ${pretrained_model_dir}/${pretrained_model_name} ]; then
        echo "Downloading pretrained TTS model..."
        local/pretrained_model_download.sh ${pretrained_model_dir} ${pretrained_model_name}
    fi
    echo "Pretrained TTS model exists: ${pretrained_model_name}"

    if [ ! -d ${voc_expdir} ]; then
        echo "Downloading pretrained PWG model..."
        local/pretrained_model_download.sh ${pretrained_model_dir} pwg_task2
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

    local/data_prep_task2.sh ${db_root} data/${org_set} ${lang} ${spk} ${trans_type}
    utils/data/resample_data_dir.sh ${fs} data/${org_set} # Downsample to fs from 24k
    utils/fix_data_dir.sh data/${org_set}
    utils/validate_data_dir.sh --no-feats data/${org_set}
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    # Trim silence parts at the beginning and the end of audio
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

    # use pretrained model cmvn
    cmvn=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "cmvn.ark" | head -n 1)

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp ${cmvn} exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp ${cmvn} exp/dump_feats/${dev_set} ${feat_dt_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    dict=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "*_units.txt" | head -n 1)
    nlsyms=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "*_non_lang_syms.txt" | head -n 1)
    echo "dictionary: ${dict}"

    # make json labels using pretrained model dict
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms "${nlsyms}" --trans_type ${trans_type} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms "${nlsyms}" --trans_type ${trans_type} \
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
        local/update_json.sh ${dumpdir}/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
fi

# add pretrained model info in config
pretrained_model_path=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "snapshot*" | head -n 1)
if [ -z "$pretrained_model_path" ]; then
    pretrained_model_path=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "model.loss*" | head -n 1)
fi
if [ -z "$pretrained_model_path" ]; then
    pretrained_model_path=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "model.last*" | head -n 1)
fi
if [ -z "$pretrained_model_path" ]; then
    echo "Cannot find pretrained model"
    exit 1
fi

train_config="$(change_yaml.py -a pretrained-model="${pretrained_model_path}" \
    -o "conf/$(basename "${train_config}" .yaml).${pretrained_model_name}.yaml" "${train_config}")"

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model fine-tuning"

    mkdir -p ${expdir}

    # copy x-vector into expdir
    # empty the scp file
    xvec_dir=exp/xvector_nnet_1a/xvectors_${train_set}
    cp ${xvec_dir}/spk_xvector.ark ${expdir}
    sed "s~${xvec_dir}/~~" ${xvec_dir}/spk_xvector.scp > ${expdir}/spk_xvector.scp

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

outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding, Synthesis"

    pids=() # initialize pids
    for name in ${dev_set}; do
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

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis"

    [ -z "${voc_checkpoint}" ] && voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t | head -n 1)"

    pids=() # initialize pids
    for name in ${dev_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        # use pretrained model cmvn
        cmvn=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "cmvn.ark" | head -n 1)
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

# Cascade ASR + TTS

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Prefinetuned model download"

    if [ ! -d ${pretrained_model_dir}/${finetuned_model_name} ]; then
        echo "Downloading finetuned TTS model..."
        local/pretrained_model_download.sh ${pretrained_model_dir} ${finetuned_model_name}
    fi
    echo "Finetuned TTS model downloaded: ${finetuned_model_name}"
fi

pairname=${srcspk}_${trgspk}_eval
expdir=exp/${srcspk}_${test_name}
[ ! -e ${expdir} ] && mkdir -p ${expdir}
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: Recognize the eval set"

    local/recognize.sh --nj ${nj} \
        --db_root ${db_root} \
        --backend pytorch \
        --api v1 \
        exp/${asr_model}_asr \
        ${expdir} \
        ${db_root}/${srcspk} \
        ${srcspk} \
        ${test_list_file}

fi

if [ -z ${tts_model_dir} ]; then
    echo "Please specify tts_model_dir!"
    exit 1
fi
outdir=${expdir}/$(basename ${tts_model_dir})_${model}; mkdir -p ${outdir}
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: Decoding"

    echo "Data preparation (cleaning text from ASR results) ..."
    tts_datadir=${expdir}/data_tts/${trgspk}; mkdir -p ${tts_datadir}
    text=${tts_datadir}/text
    local/clean_text_asr_result.py \
        ${expdir}/result/hyp.wrd.trn \
        --lang_tag en_US \
        --trans_type $trans_type > ${text}
    sed -i "s~${srcspk}_~${srcspk}_${trgspk}_~g" ${text}

    echo "Json file preparation ..."
    dict=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "*_units.txt" | head -n 1)
    nlsyms=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "*_non_lang_syms.txt" | head -n 1)
    cp ${expdir}/data_asr/utt2spk ${tts_datadir}/utt2spk # data2json.sh needs utt2spk
    sed -i "s~${srcspk}~${trgspk}~g" ${tts_datadir}/utt2spk
    sed -i "s~${trgspk}_~${srcspk}_${trgspk}_~g" ${tts_datadir}/utt2spk
    data2json.sh --nlsyms "${nlsyms}" --trans_type ${trans_type} \
         ${tts_datadir} ${dict} > ${tts_datadir}/data.json

    # use the avg x-vector in target speaker training set
    echo "Updating x vector..."
    sed "s~spk~${tts_model_dir}/spk~g" ${tts_model_dir}/spk_xvector.scp > ${tts_datadir}/spk_xvector.scp
    x_vector_ark=$(awk -v spk=$spk '/spk/{print $NF}' ${tts_datadir}/spk_xvector.scp)
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
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "stage 13: Synthesis"

    [ -z "${voc_checkpoint}" ] && voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t | head -n 1)"

    [ ! -e ${outdir}_denorm/${pairname} ] && mkdir -p ${outdir}_denorm/${pairname}
    # Denormalizing use pretrained model cmvn
    cmvn=$(find ${pretrained_model_dir}/${pretrained_model_name} -name "cmvn.ark" | head -n 1)
    apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
        scp:${outdir}/${pairname}/feats.scp \
        ark,scp:${outdir}_denorm/${pairname}/feats.ark,${outdir}_denorm/${pairname}/feats.scp

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
            ${outdir}_denorm/${pairname}/ \
            ${outdir}_denorm/${pairname}/log \
            ${outdir}_denorm/${pairname}/wav
    # PWG
    elif [ ${voc} = "PWG" ]; then
        echo "Using Parallel WaveGAN vocoder."

        # variable settings
        voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
        voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
        wav_dir=${outdir}_denorm/${pairname}/pwg_wav
        hdf5_norm_dir=${outdir}_denorm/${pairname}/hdf5_norm
        [ ! -e "${wav_dir}" ] && mkdir -p ${wav_dir}
        [ ! -e ${hdf5_norm_dir} ] && mkdir -p ${hdf5_norm_dir}

        # normalize and dump them
        echo "Normalizing..."
        ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
            parallel-wavegan-normalize \
                --skip-wav-copy \
                --config "${voc_conf}" \
                --stats "${voc_stats}" \
                --feats-scp "${outdir}_denorm/${pairname}/feats.scp" \
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
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ] ; then
    echo "stage 14: Objective Evaluation: MCD"

    minf0=$(awk '{print $1}' conf/${trgspk}.f0)
    maxf0=$(awk '{print $2}' conf/${trgspk}.f0)
    mcd_file=${outdir}_denorm/${pairname}/mcd.log

    # Decide wavdir depending on vocoder
    if [ -n "${voc}" ]; then
        # select vocoder type (GL, PWG)
        if [ ${voc} == "PWG" ]; then
            wavdir=${outdir}_denorm/${pairname}/pwg_wav
        elif [ ${voc} == "GL" ]; then
            wavdir=${outdir}_denorm/${pairname}/wav
        else
            echo "Vocoder type other than GL, PWG is not supported!"
            exit 1
        fi
    else
        echo "Please specify vocoder."
        exit 1
    fi

    ${decode_cmd} ${mcd_file} \
        mcd_calculate.py \
            --wavdir ${wavdir} \
            --gtwavdir ${db_root}/${trgspk} \
            --mcep_dim 34 \
            --shiftms 5 \
            --f0min ${minf0} \
            --f0max ${maxf0}
    grep 'Mean' ${mcd_file}

    local/ob_eval/evaluate.sh --nj ${nj} \
        --db_root ${db_root} \
        --asr_model_dir exp/${asr_model}_asr \
        ${outdir} ${pairname} ${srcspk} ${trgspk} ${wavdir}
fi
