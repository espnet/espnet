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
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# fbank feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# face feature extraction related
fps=50
lip_width=128
lip_height=64
shape_predictor_path=downloads/shape_predictor_68_face_landmarks.dat

# config files
train_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage
model=                      # VC Model checkpoint for decoding. If not specified, automatically set to the latest checkpoint 
voc=PWG                     # vocoder used (GL or PWG)
griffin_lim_iters=64        # The number of iterations of Griffin-Lim

# pretrained model related
pretrained_model=          
pretrained_tts_model_path=
#pretrained_tts_model_path=downloads/snapshot.ep.290      t

# dataset configuration
db_root=downloads
norm_name=                  # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization

# exp tag
tag="tmsv_tts_dec_unfreezed"  # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
eval_set=test


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root}/TMSV data/all
    # make train, dev and eval sets
    utils/subset_data_dir.sh --utt-list data/all/train_utt_list data/all data/${train_set}
    utils/fix_data_dir.sh data/${train_set}
    utils/subset_data_dir.sh --utt-list data/all/dev_utt_list data/all data/${dev_set}
    utils/fix_data_dir.sh data/${dev_set}
    utils/subset_data_dir.sh --utt-list data/all/eval_utt_list data/all data/${eval_set}
    utils/fix_data_dir.sh data/${eval_set}
    # the utils/subset_data_dir.sh do not split the video.scp file for us, so we need to do this seperately
    utils/filter_scp.pl data/${train_set}/utt2spk <data/all/video.scp >data/${train_set}/video.scp
    utils/filter_scp.pl data/${dev_set}/utt2spk <data/all/video.scp >data/${dev_set}/video.scp
    utils/filter_scp.pl data/${eval_set}/utt2spk <data/all/video.scp >data/${eval_set}/video.scp
fi

#if [ -z ${norm_name} ]; then
#    echo "Please specify --norm_name ."
#    exit 1
#fi
feat_tr_dir=${dumpdir}/${train_set}_fbank; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}_fbank; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}_fbank; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Fbank Feature Generation"
   
    # Generate the fbank features; by default 80-dimensional on each frame
    fbankdir=fbank

    for x in ${dev_set} ${eval_set} ${train_set}; do
        
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${x} \
            exp/make_fbank/${x} \
            ${fbankdir}

    done
        
    # compute statistics for global mean-variance normalization for fbank
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/fbank_cmvn.ark
    fbank_cmvn=data/${train_set}/fbank_cmvn.ark
    
    # dump features
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp ${fbank_cmvn} exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp ${fbank_cmvn} exp/dump_feats/${dev_set} ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp ${fbank_cmvn} exp/dump_feats/${eval_set} ${feat_ev_dir}
    echo "fbank generation and normalization succeed."

fi

face_feat_tr_dir=${dumpdir}/${train_set}_face; mkdir -p ${feat_tr_dir}
face_feat_dt_dir=${dumpdir}/${dev_set}_face; mkdir -p ${feat_dt_dir}
face_feat_ev_dir=${dumpdir}/${eval_set}_face; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Face Feature Generation"

    face_feature_dir=face_feature
    for x in ${dev_set} ${eval_set} ${train_set}; do

        make_face.sh --cmd "${train_cmd}" --nj ${nj} \
            --fps ${fps} \
            --lip_width ${lip_width} \
            --lip_height ${lip_height} \
            --shape_predictor_path ${shape_predictor_path} \
            data/${x} \
            exp/make_face/${x} \
            ${face_feature_dir}

    done
            
    # compute statistics for global mean-variance normalization for fbank
    compute-cmvn-stats scp:data/${train_set}/face_feats.scp data/${train_set}/face_cmvn.ark
    face_cmvn=data/${train_set}/face_cmvn.ark
    
    # dump features
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${train_set} ${face_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${dev_set} ${face_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${eval_set} ${face_feat_ev_dir}
    echo "face-feature generation and normalization succeed."

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Dictionary and Json Data Preparation"

    # make dummy dict
    dict="data/dummy_dict/X.txt"
    if [ ! -e ${dict} ]; then
        mkdir -p ${dict%/*}
        echo "<unk> 1" > ${dict}
    fi
    
    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${face_feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${face_feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${face_feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${face_feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
    data2json.sh --feat ${face_feat_ev_dir}/feats.scp \
         data/${eval_set} ${dict} > ${face_feat_ev_dir}/data.json
fi
pair_tr_dir=dump/${train_set}; mkdir -p ${pair_tr_dir}
pair_dt_dir=dump/${dev_set}; mkdir -p ${pair_dt_dir}
pair_ev_dir=dump/${eval_set}; mkdir -p ${pair_ev_dir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Pair Json Data Preparation"

    # make pair json
    pair_face_and_fbank_json.py \
        --face-json ${face_feat_tr_dir}/data.json \
        --fbank-json ${feat_tr_dir}/data.json \
        -O ${pair_tr_dir}/data.json
    pair_face_and_fbank_json.py \
        --face-json ${face_feat_dt_dir}/data.json \
        --fbank-json ${feat_dt_dir}/data.json \
        -O ${pair_dt_dir}/data.json
    pair_face_and_fbank_json.py \
        --face-json ${face_feat_ev_dir}/data.json \
        --fbank-json ${feat_ev_dir}/data.json \
        -O ${pair_ev_dir}/data.json
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in ${train_set} ${dev_set} ${eval_set}; do
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
        utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
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
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
    # Update json
    for name in ${train_set} ${dev_set} ${eval_set}; do
        local/update_json.sh ${dumpdir}/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
fi


if [[ -z ${train_config} ]]; then
    echo "Please specify --train_config."
    exit 1
fi

# If pretrained model specified, add pretrained model info in config
if [ -n "${pretrained_tts_model_path}" ]; then
    train_config="$(change_yaml.py \
        -a pretrained-tts-model="${pretrained_tts_model_path}" \
        -a encoder-reduction-factor=2 \
        -o "conf/$(basename "${train_config}" .yaml).${tag}.yaml" "${train_config}")"
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi


expdir=exp/${expname}
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: LTS model training"

    mkdir -p ${expdir}
    tr_json=${pair_tr_dir}/data.json
    dt_json=${pair_dt_dir}/data.json

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        lts_train.py \
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
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding"

    echo "Decoding..."
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            lts_decode.py \
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
    
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Stage 8: Synthesis"

    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        
        # Normalization
        # If not using pretrained models statistics, use statistics of target speaker
        if [ -n "${pretrained_model}" ]; then
            fbank_cmvn="$(find "${db_root}/${pretrained_model}" -name "cmvn.ark" -print0 | xargs -0 ls -t | head -n 1)"
        else
            fbank_cmvn=data/${train_set}/fbank_cmvn.ark
        fi
        apply-cmvn --norm-vars=true --reverse=true ${fbank_cmvn} \
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
            voc_expdir=${db_root}/pwg
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



if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "Stage 9 MCD evaluation for output"
    mcep_dim=24
    shift_ms=5
    num_of_spks=18
    for count_of_spk in $(seq 1 1 $num_of_spks); do
        spk=SP$(printf "%02d" $count_of_spk)

        out_wavdir=${outdir}_denorm/${eval_set}/pwg_wav
        gt_wavdir=${db_root}/TMSV/${spk}/audio
        minf0=$(awk '{print $1}' ${db_root}/TMSV/conf/${spk}.f0)
        echo "$minf0"
        maxf0=$(awk '{print $2}' ${db_root}/TMSV/conf/${spk}.f0)
        echo "$maxf0"
        out_spk_wavdir=${outdir}_denorm.ob_eval/mcd_eval/pwg_out/${spk}
        gt_spk_wavdir=${outdir}_denorm.ob_eval/mcd_eval/gt/${spk}
        mcd_file=${outdir}_denorm.ob_eval/mcd_eval/${spk}_pwg_mcd.log
        mkdir -p ${out_spk_wavdir}
        mkdir -p ${gt_spk_wavdir}
        
        local/make_spk_dir_for_mcd_eval.sh ${out_wavdir} ${gt_wavdir} \
            ${out_spk_wavdir} ${gt_spk_wavdir}
        ${decode_cmd} ${mcd_file} \
            mcd_calculate.py \
                --wavdir ${out_spk_wavdir} \
                --gtwavdir ${gt_spk_wavdir} \
                --mcep_dim ${mcep_dim} \
                --shiftms ${shift_ms} \
                --f0min ${minf0} \
                --f0max ${maxf0}
           
    done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "Stage 10: objective evaluation on ASR"
    local/ob_eval/evaluate.sh ${outdir} ${eval_set}
fi
