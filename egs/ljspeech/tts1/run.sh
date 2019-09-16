#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#           2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=6
stop_stage=6
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=22050      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# feature configuration
do_delta=false

# config files
train_config=$1 # you can select from conf or conf/tuning.
                                               # now we support tacotron2, transformer, and fastspeech
                                               # see more info in the header of each config.
decode_config=conf/decode.yaml

# decoding related
model=model.last1.avg.best #model.loss.best
n_average=0 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# objective evaluation related
asr_model="librispeech.transformer.ngpu4"

# root directory of db
db_root=/abelab/DB4 #downloads

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Name sub-sets
train_set="train_no_dev"
dev_set="dev"
eval_set="eval"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root}/LJSpeech-1.1 data/train
    utils/validate_data_dir.sh --no-feats data/train
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
        data/train \
        exp/make_fbank/train \
        ${fbankdir}

    # make a dev set
    utils/subset_data_dir.sh --last data/train 500 data/deveval
    utils/subset_data_dir.sh --last data/deveval 250 data/${eval_set}
    utils/subset_data_dir.sh --first data/deveval 250 data/${dev_set}
    n=$(( $(wc -l < data/train/wav.scp) - 500 ))
    utils/subset_data_dir.sh --first data/train ${n} data/${train_set}

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
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
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Text-to-speech model training"
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
    echo "stage 6: Objective Evaluation"

    # ASR model selection for CER objective evaluation 
    asr_model_dir="exp/${asr_model}_asr"
    case "${asr_model}" in
        "librispeech.transformer.ngpu1") asr_url="https://drive.google.com/open?id=1bOaOEIZBveERti0x6mnBYiNsn6MSRd2E" \
          asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
          asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer.yaml" \ 
          recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/results/model.last10.avg.best" \
          lang_model="${asr_model_dir}/exp/train_rnnlm_pytorch_lm_unigram5000/rnnlm.model.best" ;;
    
        "librispeech.transformer.ngpu4") asr_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" \
          asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
          asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer_large.yaml" \
          recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best" \
          lang_model="${asr_model_dir}/exp/irielm.ep11.last5.avg/rnnlm.model.best" ;;        
        
        *) echo "No such models: ${asr_model}"; exit 1 ;;
    esac

    # ASR model download (librispeech)
    if [ ! -e ${asr_model_dir}/.complete ]; then
        mkdir -p ${asr_model_dir}
        download_from_google_drive.sh ${asr_url} ${asr_model_dir} ".tar.gz"
        touch ${asr_model_dir}/.complete
    fi
    echo "ASR model: ${asr_model_dir} exits."

    # Select decoder
    voc="GL"
    if [ ${voc} == "GL" ]; then
        dir_tail=""
    elif [ ${voc} == "WNV_softmax" ]; then
        dir_tail="_wnv_nsf"
    elif [ ${voc} == "WNV_mol" ]; then
        dir_tail="_wnv_mol"
    fi

    asr_data_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.data${dir_tail}"
    asr_fbank_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.fbank${dir_tail}"
    asr_feat_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.dump${dir_tail}"
    asr_result_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.result${dir_tail}"

    # Data preparation for ASR
    echo "6.1 Data preparation for ASR"
    for name in ${dev_set} ${eval_set}; do
        local/data_prep_for_asr.sh ${outdir}_denorm/${name}/wav${dir_tail} ${asr_data_dir}/${name}
        cp data/${name}/text ${asr_data_dir}/${name}/text
        utils/validate_data_dir.sh --no-feats ${asr_data_dir}/${name}
    done
    
    # Feature extraction for ASR
    echo "6.2 Feature extraction for ASR"
    for name in ${dev_set} ${eval_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
          --write_utt2num_frames true \
          --write_utt2dur false \
          ${asr_data_dir}/${name} \
          ${outdir}_denorm.ob_eval/${asr_model}_asr.make_fbank${dir_tail}/${name} \
          ${asr_fbank_dir}/${name}
        utils/fix_data_dir.sh ${asr_data_dir}/${name}

        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
          ${asr_data_dir}/${name}/feats.scp ${asr_cmvn} ${outdir}_denorm.ob_eval/${asr_model}_asr.dump_feats${dir_tail}/${name} \
          ${asr_feat_dir}/${name}
    done

    # Dictionary and Json Data Preparation
    echo "6.3 Dictionary and Json Data Preparation"
    asr_dict="data/decode_dict/X.txt"; mkdir -p ${asr_dict%/*}
    echo "<unk> 1" > ${asr_dict}
    for name in ${dev_set} ${eval_set}; do
        data2json.sh --feat ${asr_feat_dir}/${name}/feats.scp \
          ${asr_data_dir}/${name} ${asr_dict} > ${asr_feat_dir}/${name}/data.json
    done
    
    # ASR decoding
    echo "6.4 ASR decoding"
    asr_decode_config="conf/tuning/decode_asr.yaml"
    cat ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}
    for name in ${dev_set} ${eval_set}; do

        # split data
        splitjson.py --parts ${nj} ${asr_feat_dir}/${name}/data.json
    
        # set batchsize 0 to disable batch decoding    
        ${decode_cmd} JOB=1:${nj} ${asr_result_dir}/${name}/log/decode.JOB.log \
            asr_recog.py \
              --config ${asr_decode_config} \
              --ngpu 0 \
              --backend ${backend} \
              --batchsize 0 \
              --recog-json ${asr_feat_dir}/${name}/split${nj}utt/data.JOB.json \
              --result-label ${asr_result_dir}/${name}/data.JOB.json \
              --model ${recog_model} \
              --api v2 \
              --rnnlm ${lang_model}

        score_sclite_wo_dict.sh --wer true ${asr_result_dir}/${name}

    done

fi

