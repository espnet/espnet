#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# genearl configuration
stage=-1
ngpu=1       # number of gpu in training
nj=32        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)
# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=512   # number of shift points
win_length="" # window length
# encoder related
embed_dim=512
elayers=1
eunits=512
econv_layers=3 # if set 0, no conv layer is used
econv_chans=512
econv_filts=5
# decoder related
spk_embed_dim=512
dlayers=2
dunits=1024
prenet_layers=2  # if set 0, no prenet is used
prenet_units=256
postnet_layers=5 # if set 0, no postnet is used
postnet_chans=512
postnet_filts=5
# attention related
adim=128
aconv_chans=32
aconv_filts=15      # resulting in filter_size = aconv_filts * 2 + 1
cumulate_att_w=true # whether to cumulate attetion weight
use_batch_norm=true # whether to use batch normalization in conv layer
use_concate=true    # whether to concatenate encoder embedding with decoder lstm outputs
use_residual=false  # whether to use residual connection in encoder convolution
use_masking=true    # whether to mask the padded part in loss calculation
bce_pos_weight=1.0  # weight for positive samples of stop token in cross-entropy calculation
# minibatch related
batchsize=32
batch_sort_key="output" # empty or input or output (if empty, shuffled batch will be used)
maxlen_in=150           # if input length  > maxlen_in, batchsize is reduced (if batch_sort_key="", not effect)
maxlen_out=400          # if output length > maxlen_out, batchsize is reduced (if batch_sort_key="", not effect)
# optimization related
lr=1e-3
eps=1e-6
weight_decay=0.0
dropout=0.5
zoneout=0.1
epochs=50
# decoding related
model=model.loss.best
threshold=0.7    # threshold to stop the generation
maxlenratio=10.0 # maximum length of generated samples = input length * maxlenratio
minlenratio=0.0  # minimum length of generated samples = input length * minlenratio

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_clean_460
train_dev=dev
eval_set=test_clean

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
    done
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    fbankdir=fbank
    for x in dev_clean test_clean train_clean_100 train_clean_360; do
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

    utils/combine_data.sh data/${train_set}_org data/train_clean_100 data/train_clean_360
    utils/combine_data.sh data/${train_dev}_org data/dev_clean

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
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
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
fi

if [ ${stage} -le 3 ]; then
    echo "stage 3: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in ${train_set} ${train_dev} ${eval_set}; do
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc exp/make_mfcc $mfccdir
        utils/fix_data_dir.sh data/${name}_mfcc
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc
    done
    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e $nnet_dir ];then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # Extract x-vector
    for name in ${train_set} ${train_dev} ${eval_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            $nnet_dir data/${name}_mfcc \
            $nnet_dir/xvectors_${name}
    done
    # Update json
    for name in ${train_set} ${train_dev} ${eval_set}; do
        local/update_json.sh ${dumpdir}/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
fi


if [ -z ${tag} ];then
    expdir=exp/${train_set}_taco2_enc${embed_dim}
    if [ ${econv_layers} -gt 0 ];then
        expdir=${expdir}-${econv_layers}x${econv_filts}x${econv_chans}
    fi
    expdir=${expdir}-${elayers}x${eunits}_dec${dlayers}x${dunits}
    if [ ${prenet_layers} -gt 0 ];then
        expdir=${expdir}_pre${prenet_layers}x${prenet_units}
    fi
    if [ ${postnet_layers} -gt 0 ];then
        expdir=${expdir}_post${postnet_layers}x${postnet_filts}x${postnet_chans}
    fi
    expdir=${expdir}_att${adim}-${aconv_filts}x${aconv_chans}
    if ${cumulate_att_w};then
        expdir=${expdir}_cm
    fi
    if ${use_batch_norm};then
        expdir=${expdir}_bn
    fi
    if ${use_residual};then
        expdir=${expdir}_rs
    fi
    if ${use_concate};then
        expdir=${expdir}_cc
    fi
    if ${use_masking};then
        expdir=${expdir}_msk_pw${bce_pos_weight}
    fi
    expdir=${expdir}_do${dropout}_zo${zoneout}_lr${lr}_ep${eps}_wd${weight_decay}_bs$((batchsize*ngpu))
    if [ ! -z ${batch_sort_key} ];then
        expdir=${expdir}_sort_by_${batch_sort_key}_mli${maxlen_in}_mlo${maxlen_out}
    fi
    expdir=${expdir}_sd${seed}
else
    expdir=exp/${train_set}_${tag}
fi
if [ ${stage} -le 4 ];then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --embed_dim ${embed_dim} \
           --elayers ${elayers} \
           --eunits ${eunits} \
           --econv_layers ${econv_layers} \
           --econv_chans ${econv_chans} \
           --econv_filts ${econv_filts} \
           --spk_embed_dim ${spk_embed_dim} \
           --dlayers ${dlayers} \
           --dunits ${dunits} \
           --prenet_layers ${prenet_layers} \
           --prenet_units ${prenet_units} \
           --postnet_layers ${postnet_layers} \
           --postnet_chans ${postnet_chans} \
           --postnet_filts ${postnet_filts} \
           --adim ${adim} \
           --aconv-chans ${aconv_chans} \
           --aconv-filts ${aconv_filts} \
           --cumulate_att_w ${cumulate_att_w} \
           --use_batch_norm ${use_batch_norm} \
           --use_concate ${use_concate} \
           --use_residual ${use_residual} \
           --use_masking ${use_masking} \
           --bce_pos_weight ${bce_pos_weight} \
           --lr ${lr} \
           --eps ${eps} \
           --dropout-rate ${dropout} \
           --zoneout-rate ${zoneout} \
           --weight-decay ${weight_decay} \
           --batch_sort_key ${batch_sort_key} \
           --batch-size ${batchsize} \
           --maxlen-in ${maxlen_in} \
           --maxlen-out ${maxlen_out} \
           --epochs ${epochs}
fi

outdir=${expdir}/outputs_${model}_th${threshold}_mlr${minlenratio}-${maxlenratio}
if [ ${stage} -le 5 ];then
    echo "stage 5: Decoding"
    for sets in ${train_dev} ${eval_set};do
        [ ! -e  ${outdir}/${sets} ] && mkdir -p ${outdir}/${sets}
        cp ${dumpdir}/${sets}/data.json ${outdir}/${sets}
        splitjson.py --parts ${nj} ${outdir}/${sets}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:$nj ${outdir}/${sets}/log/decode.JOB.log \
            tts_decode.py \
                --backend pytorch \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${sets}/feats.JOB \
                --json ${outdir}/${sets}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --model-conf ${expdir}/results/model.conf \
                --threshold ${threshold} \
                --maxlenratio ${maxlenratio} \
                --minlenratio ${minlenratio}
        # concatenate scp files
        for n in $(seq $nj); do
            cat "${outdir}/${sets}/feats.$n.scp" || exit 1;
        done > ${outdir}/${sets}/feats.scp
    done
fi

if [ ${stage} -le 6 ];then
    echo "stage 6: Synthesis"
    for sets in ${train_dev} ${eval_set};do
        [ ! -e ${outdir}_denorm/${sets} ] && mkdir -p ${outdir}_denorm/${sets}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${sets}/feats.scp \
            ark,scp:${outdir}_denorm/${sets}/feats.ark,${outdir}_denorm/${sets}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            ${outdir}_denorm/${sets} \
            ${outdir}_denorm/${sets}/log \
            ${outdir}_denorm/${sets}/wav
    done
fi
