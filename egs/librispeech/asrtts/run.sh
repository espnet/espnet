#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false
# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# network architecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=5
eunits=1024
eprojs=1024
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=1024
# attention related
atype=location
adim=100
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.0

# minibatch related
batchsize=20
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# other config
drop=0.2

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
opt=adadelta
epochs=10
patience=3

# rnnlm related
lm_layers=1
lm_units=1024
lm_opt=sgd        # or adam
lm_sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
lm_batchsize=1024 # batch size in LM training
lm_epochs=20      # if the data size is large, we can reduce this
lm_patience=3
lm_maxlen=40      # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
lm_weight=0.0
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.0
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.5

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data
datadir=/mnt/matylda2/data/librispeech_kaldi_download

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
use_bpe=false

# training related
asr_train=true
asr_decode=true
tts_train=true
tts_decode=true
asrtts_train=true
asrtts_decode=true
unpair=dualp
policy_gradient=true
use_rnnlm=false
rnnlm_loss=none

# exp tag
tag="" # tag for managing experiments.
asr_model_conf=$PWD/pretrained_models/librispeech_100/asr/results/model.json
asr_model=$PWD/pretrained_models/librispeech_100/asr/results/model.acc.best
rnnlm_model=$PWD/rnnlm_models/librispeech_360/rnnlm.model.best
rnnlm_model_conf=$PWD/rnnlm_models/librispeech_360/model.json
tts_model=$PWD/pretrained_models/librispeech_100/tts/results/model.loss.best
tts_model_conf=$PWD/pretrained_models/librispeech_100/tts/results/model.json
spk_vector=exp/xvector_nnet_1a

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_paired_set=train_clean_100
train_unpaired_set=train_clean_360
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"

echo "stage -3: Data Download"
for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    if [ ! -s ${datadir}/${part}.tar.gz ]; then
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    fi
done

### Task dependent. You have to make data the following preparation part by yourself.
### But you can utilize Kaldi recipes in most cases
echo "stage -2: Data preparation"
for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    if [ ! -s data/${part//-/_}/wav.scp ]; then
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    fi
done

nj=20
dev_set=$train_dev
eval_set=test_clean
fbankdir=fbank
feat_tr_dir=${dumpdir}/${train_set}; 
feat_tr_p_dir=${dumpdir}/${train_paired_set};
feat_tr_up_dir=${dumpdir}/${train_unpaired_set};
feat_dt_dir=${dumpdir}/${dev_set};
feat_ev_dir=${dumpdir}/${eval_set};
dict=data/lang_char/${train_set}_units.txt
nlsyms=data/lang_char/non_lang_syms.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
scratch=/mnt/scratch06/tmp/baskar/espnet_new/features
nnet_dir=exp/xvector_nnet_1a

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Tmp space allocation"
    local/make_symlink_dir.sh --tmp-root $scratch/$fbankdir $fbankdir
    local/make_symlink_dir.sh --tmp-root $scratch/$feat_tr_dir $feat_tr_dir
    local/make_symlink_dir.sh --tmp-root $scratch/$feat_tr_p_dir $feat_tr_p_dir
    local/make_symlink_dir.sh --tmp-root $scratch/$feat_tr_up_dir $feat_tr_up_dir
    local/make_symlink_dir.sh --tmp-root $scratch/$feat_dt_dir $feat_dt_dir
    local/make_symlink_dir.sh --tmp-root $scratch/$feat_ev_dir $feat_ev_dir
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Feature extraction for TTS and ASR"
    for x in dev_clean test_clean train_clean_100 train_clean_360; do
        if [ ! -s data/${x}/feats.scp ]; then
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
        fi
    done
    utils/combine_data.sh data/${train_set}_org data/train_clean_100 data/train_clean_360
    utils/combine_data.sh data/${dev_set}_org data/dev_clean
    utils/copy_data_dir.sh data/${train_paired_set} data/${train_paired_set}_org
    utils/copy_data_dir.sh data/${train_unpaired_set} data/${train_unpaired_set}_org
    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${dev_set}_org data/${dev_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_paired_set}_org data/${train_paired_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_unpaired_set}_org data/${train_unpaired_set}
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_paired_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train_p ${feat_tr_p_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_unpaired_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train_up ${feat_tr_up_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: JSON preparation for TTS and ASR"
    echo "dictionary: ${dict}"
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    if [ $use_bpe == 'true' ]; then
        cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
        spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
        spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    else
        echo "make a non-linguistic symbol list"
        #cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
        #cat ${nlsyms}
        text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    fi
    wc -l ${dict}
    # make json labels
    if [ ! -s ${feat_ev_dir}/data.json ]; then
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_tr_p_dir}/feats.scp \
         data/${train_paired_set} ${dict} > ${feat_tr_p_dir}/data.json
    data2json.sh --feat ${feat_tr_up_dir}/feats.scp \
         data/${train_unpaired_set} ${dict} > ${feat_tr_up_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
    fi
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in ${train_set} ${dev_set} ${eval_set}; do
        if [ ! -s data/${name}_mfcc/feats.scp ]; then
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc
        steps/make_mfcc.sh \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc exp/make_mfcc ${mfccdir}
        utils/fix_data_dir.sh data/${name}_mfcc
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc
        fi
    done
    # Check pretrained model existence
    if [ ! -e ${nnet_dir} ];then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # Extract x-vector
    for name in ${train_set} ${dev_set} ${eval_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/${name}_mfcc \
            ${nnet_dir}/xvectors_${name}
    done
    # Update json
    for name in ${train_set} ${dev_set} ${eval_set}; do
        local/update_json.sh ${dumpdir}/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: LM training"
    if [ $use_bpe == 'true' ]; then
        lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
        lmexpdir=exp/${lmexpname}
        mkdir -p ${lmexpdir}
        lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    elif [ $use_bpe == 'false' ]; then
        lmexpname=train_rnnlm_${backend}_${lmtag}_char
        lmexpdir=exp/${lmexpname}
        mkdir -p ${lmexpdir}
        lmdatadir=data/local/lm_train_char
    fi
    mkdir -p ${lmdatadir}
    # use external data
    if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
    fi
    cut -f 2- -d" " data/${train_set}/text | gzip -c > data/local/lm_train/${train_set}_text.gz
    if [ $use_bpe == 'true' ]; then
        # combine external text and transcriptions and shuffle them with seed 777
        zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
            spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
            > ${lmdatadir}/valid.txt
    else
        echo "char text preparation for LM"
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        zcat data/local/lm_train/librispeech-lm-norm.txt.gz \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${eval_set}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. single gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --sortagrad ${lm_sortagrad} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --patience ${lm_patience} \
        --maxlen ${lm_maxlen} \
        --dict ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: ASR training and decode"
    expdir=exp/asr_${tag}
    expname=asr_${tag}
    if [ $asr_train == 'true' ]; then
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_p_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --dropout-rate ${drop} \
        --opt ${opt} \
        --sortagrad ${sortagrad} \
        --epochs ${epochs} \
        --patience ${patience}

    fi
    if [ $asr_decode == 'true' ]; then
    nj=32

    pids=() # initialize pids
    for rtask in ${eval_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
        #### use CPU for decoding
        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu 0 \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} 

        if [ $use_bpe == 'true' ]; then
            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        else
            score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: TTS training, decode and synthesize"
    ttsexpdir=exp/tts_${tag}
    tr_json=$feat_tr_p_dir/data_tts.json
    dt_json=$feat_dt_dir/data_tts.json
    seed=1
    # decoding related
    model=model.loss.best
    threshold=0.5    # threshold to stop the generation
    maxlenratio=10.0 # maximum length of generated samples = input length * maxlenratio
    minlenratio=0.0  # minimum length of generated samples = input length * minlenratio
    train_config=conf/train_pytorch_tacotron2+spkemb.yaml
    #decode_config=conf/decode.yaml

    for name in ${train_paired_set} ${dev_set}; do
        cp ${dumpdir}/${name}/data.json ${dumpdir}/${name}/data_tts.json
        if [ $name == ${train_paired_set} ]; then fname=${train_set}; else fname=$name; fi
        local/update_json.sh ${dumpdir}/${name}/data_tts.json ${nnet_dir}/xvectors_${fname}/xvector.scp
    done


    if [ $tts_train == 'true' ]; then
    ${cuda_cmd} --gpu 1 ${ttsexpdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu 1 \
           --config $train_config \
           --outdir ${ttsexpdir}/results \
           --tensorboard-dir tensorboard/${ttsexpdir} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} 
        fi
    if [ $tts_decode == 'true' ]; then
    outdir=${ttsexpdir}/outputs_${model}_th${threshold}_mlr${minlenratio}-${maxlenratio}
    for name in ${dev_set} ${eval_set};do
        [ ! -e  ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
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
                --model ${ttsexpdir}/results/${model} \
                --threshold ${threshold} \
                --maxlenratio ${maxlenratio} \
                --minlenratio ${minlenratio}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    done
    for name in ${dev_set} ${eval_set};do
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
            ${outdir}_denorm/${name} \
            ${outdir}_denorm/${name}/log \
            ${outdir}_denorm/${name}/wav
    done
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: ASR-TTS training, decode and synthesize"
    asrttsexpdir=exp/asrtts_${tag}
    train_opts=
    sample_topk=0
    sample_scaling=0.1
    teacher_weight=0.1
    n_samples=5

    if [ ! -s  ${feat_tr_up_dir}/data_rnd.json ]; then
        bash utils/copy_data_dir.sh data/${train_unpaired_set} data/${train_unpaired_set}_rnd
        bash local/rand_datagen.sh --jsonout "${feat_tr_up_dir}/data_rnd.json" \
            --nlsyms $nlsyms --dict $dict --xvec ${spk_vector}/xvectors_$train_set \
            $feat_tr_p_dir $feat_tr_up_dir data/${train_unpaired_set}_rnd
    fi
    if [ $unpair == 'dual' ]; then
        tr_json_list="${feat_tr_up_dir}/data_rnd.json"
    elif [ $unpair == 'dualp' ]; then
       tr_json_list="${feat_tr_up_dir}/data_rnd.json ${feat_tr_p_dir}/data.json"
    else
        tr_json_list="${feat_tr_p_dir}/data.json"
    fi
    if [ "$policy_gradient" = "true" ]; then
        asrttsexpdir=${asrttsexpdir}_exploss_pgrad
        train_opts="$train_opts --policy-gradient"
    fi
    if [ "$rnnlm_loss" = "ce" ]; then
        asrttsexpdir=${asrttsexpdir}_rnnlmloss_${rnnlm_loss}
        train_opts="$train_opts --rnnlm $rnnlm_model --rnnlm-conf $rnnlm_model_conf --rnnloss ce" 
    elif [ "$rnnlm_loss" = "kld" ]; then
        asrttsexpdir=${asrttsexpdir}_rnnlmloss_${rnnlm_loss}
        train_opts="$train_opts --rnnlm $rnnlm_model --rnnlm-conf $rnnlm_model_conf --rnnloss kld" 
    elif [ "$rnnlm_loss" = "mmd" ]; then
        asrttsexpdir=${asrttsexpdir}_rnnlmloss_${rnnlm_loss}
        train_opts="$train_opts --rnnlm $rnnlm_model --rnnlm-conf $rnnlm_model_conf --rnnloss mmd" 
    elif [ "$rnnlm_loss" = "kl" ]; then
        asrttsexpdir=${asrttsexpdir}_rnnlmloss_${rnnlm_loss}
        train_opts="$train_opts --rnnlm $rnnlm_model --rnnlm-conf $rnnlm_model_conf --rnnloss kl" 
    fi
    if [ $asrtts_train == 'true' ]; then
        ${cuda_cmd} --gpu 1 ${asrttsexpdir}/train.log \
        asrtts_train.py \
        --ngpu 6 \
        --backend ${backend} \
        --outdir ${asrttsexpdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${asrttsexpdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${tr_json_list} \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs} \
        --asr-model-conf $asr_model_conf \
        --asr-model $asr_model \
        --tts-model-conf $tts_model_conf \
        --tts-model $tts_model \
        --criterion acc \
        --update-asr-only \
        --generator tts \
        --sample-topk $sample_topk \
        --sample-scaling $sample_scaling \
        --teacher-weight $teacher_weight \
        --n-samples-per-input $n_samples \
        $train_opts
    fi
    if [ $asrtts_decode == 'true' ]; then
        nj=32
        for rtask in ${recog_set}; do
        (
            recog_opts=
            if [ $use_rnnlm = true ]; then
                decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
                recog_opts="$recog_opts --lm-weight ${lm_weight} --rnnlm ${lmexpdir}/rnnlm.model.best"
            else
                decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
            fi
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

            # split data
            splitjson.py --parts ${nj} ${feat_recog_dir}/data.json 

            #### use CPU for decoding
            ngpu=0

            ${decode_cmd} JOB=1:${nj} ${asrttsexpdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${asrttsexpdir}/${decode_dir}/data.JOB.json \
            --model ${asrttsexpdir}/results/model.${recog_model}  \
            --model-conf ${asrttsexpdir}/results/model.json  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            $recog_opts \
            &
            wait
            score_sclite.sh --wer true ${asrttsexpdir}/${decode_dir} ${dict}

        ) &
        done
        wait
        echo "Finished"
    fi
fi


