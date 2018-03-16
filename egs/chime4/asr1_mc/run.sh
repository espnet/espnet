#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
gpu=-1         # use 0 when using GPU on slurm/grid engine, otherwise -1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# loss related
ctctype=chainer
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=10
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# decoding parameter
beam_size=20
penalty=0
maxlenratio=0.8
minlenratio=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# me2e
cmvn=data-fbank/tr05_multi_noisy/cmvn.ark
melmat=exp/make_melmat/melmat.ark
trainmode="noisy+enhan" # training mode
recogmode="enhan" # decoding mode

# data
chime4_data=/export/corpora4/CHiME4/CHiME3 # JHU setup

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh 
. ./cmd.sh 

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr05_multi_noisy

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make the following data preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    wsj0_data=${chime4_data}/data/WSJ0
    local/clean_wsj0_data_prep.sh ${wsj0_data}
    local/clean_chime4_format_data.sh
    echo "beamforming for multichannel cases"
    local/run_beamform_2ch_track.sh --cmd "${train_cmd}" --nj 20 \
        ${chime4_data}/data/audio/16kHz/isolated_2ch_track enhan/beamformit_2mics
    local/run_beamform_6ch_track.sh --cmd "${train_cmd}" --nj 20 \
        ${chime4_data}/data/audio/16kHz/isolated_6ch_track enhan/beamformit_5mics
    echo "prepartion for chime4 data"
    local/real_noisy_chime4_data_prep.sh ${chime4_data}
    local/simu_noisy_chime4_data_prep.sh ${chime4_data}
    echo "test data for 1ch track"
    local/real_enhan_chime4_data_prep.sh isolated_1ch_track ${chime4_data}/data/audio/16kHz/isolated_1ch_track
    local/simu_enhan_chime4_data_prep.sh isolated_1ch_track ${chime4_data}/data/audio/16kHz/isolated_1ch_track
    echo "test data for 2ch track"
    local/real_enhan_chime4_data_prep.sh beamformit_2mics ${PWD}/enhan/beamformit_2mics
    local/simu_enhan_chime4_data_prep.sh beamformit_2mics ${PWD}/enhan/beamformit_2mics
    echo "test data for 6ch track"
    local/real_enhan_chime4_data_prep.sh beamformit_5mics ${PWD}/enhan/beamformit_5mics
    local/simu_enhan_chime4_data_prep.sh beamformit_5mics ${PWD}/enhan/beamformit_5mics
fi

if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks to obtain cmvn statistics
    fbankdir=fbank
    tasks="tr05_real_noisy tr05_simu_noisy"
    for x in ${tasks}; do
        utils/copy_data_dir.sh data/${x} data-fbank/${x}
        steps/make_fbank.sh --nj 8 --cmd "${train_cmd}" data-fbank/${x} exp/make_fbank/${x} ${fbankdir}
    done

    echo "combine real and simulation data"
    utils/combine_data.sh data-fbank/${train_set} data-fbank/tr05_simu_noisy data-fbank/tr05_real_noisy

    # compute global CMVN
    compute-cmvn-stats scp:data-fbank/${train_set}/feats.scp data-fbank/${train_set}/cmvn.ark

    # compute Mel-filterbank matrix
    fbank_config=conf/fbank.conf
    meldir=exp/make_melmat; mkdir -p ${meldir}
    melmat.py --config=${fbank_config} ${meldir}/melmat.ark

    # Generate the stft features; by default 257-dimensional stfts
    channels="1 2 3 4 5 6"

    # noisy
    tasks="\
    tr05_real_noisy tr05_simu_noisy dt05_real_noisy dt05_simu_noisy et05_real_noisy et05_simu_noisy \
    dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track \
    "
    stftdir=stft
    for x in ${tasks}; do
        utils/copy_data_dir.sh data/${x} data-stft/${x}
        local/make_stft.sh --nj 8 --cmd "${train_cmd}" data-stft/${x} exp/make_stft/${x} ${stftdir}
    done

    echo "combine real and simulation data"
    utils/combine_data.sh data-stft/tr05_multi_noisy data-stft/tr05_real_noisy data-stft/tr05_simu_noisy
    utils/combine_data.sh data-stft/dt05_multi_noisy data-stft/dt05_real_noisy data-stft/dt05_simu_noisy
    utils/combine_data.sh data-stft/dt05_multi_isolated_1ch_track data-stft/dt05_real_isolated_1ch_track data-stft/dt05_simu_isolated_1ch_track

    # enhan
    tasks="\
    tr05_real_noisy tr05_simu_noisy dt05_real_noisy dt05_simu_noisy et05_real_noisy et05_simu_noisy \
    "
    stftdir=stft
    for x in ${tasks}; do
        for ch in ${channels}; do
            utils/copy_data_dir.sh data/${x} data-stft/${x}_ch${ch}
            for f in text utt2spk wav.scp; do
                cat data-stft/${x}_ch${ch}/${f} | grep CH${ch} > data-stft/${x}_ch${ch}/${f}.tmp
                cp -a data-stft/${x}_ch${ch}/${f}.tmp data-stft/${x}_ch${ch}/${f}
                rm data-stft/${x}_ch${ch}/${f}.tmp

                sed -i -e "s/\.CH${ch}_REAL//g" data-stft/${x}_ch${ch}/${f}
                sed -i -e "s/\.CH${ch}_SIMU//g" data-stft/${x}_ch${ch}/${f}
            done
            utils/utt2spk_to_spk2utt.pl data-stft/${x}_ch${ch}/utt2spk > data-stft/${x}_ch${ch}/spk2utt
            local/make_stft.sh --nj 8 --cmd "${train_cmd}" data-stft/${x}_ch${ch} exp/make_stft/${x}_ch${ch} ${stftdir}
        done
    done

    echo "combine real and simulation data"
    for ch in ${channels}; do
        utils/combine_data.sh data-stft/tr05_multi_noisy_ch${ch} data-stft/tr05_real_noisy_ch${ch} data-stft/tr05_simu_noisy_ch${ch}
        utils/combine_data.sh data-stft/dt05_multi_noisy_ch${ch} data-stft/dt05_real_noisy_ch${ch} data-stft/dt05_simu_noisy_ch${ch}
    done

fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data-stft/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data-stft/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    # noisy
    data2json.sh --feat data-stft/tr05_multi_noisy/feats.scp --nlsyms ${nlsyms} \
         data-stft/tr05_multi_noisy ${dict} > data-stft/tr05_multi_noisy/data.json
    data2json.sh --feat data-stft/dt05_multi_noisy/feats.scp --nlsyms ${nlsyms} \
         data-stft/dt05_multi_noisy ${dict} > data-stft/dt05_multi_noisy/data.json
    data2json.sh --feat data-stft/dt05_multi_isolated_1ch_track/feats.scp --nlsyms ${nlsyms} \
         data-stft/dt05_multi_isolated_1ch_track ${dict} > data-stft/dt05_multi_isolated_1ch_track/data.json

    # enhan
    for ch in ${channels}; do
        data2json.sh --feat data-stft/tr05_multi_noisy_ch${ch}/feats.scp --nlsyms ${nlsyms} \
             data-stft/tr05_multi_noisy_ch${ch} ${dict} > data-stft/tr05_multi_noisy_ch${ch}/data.json
        data2json.sh --feat data-stft/dt05_multi_noisy_ch${ch}/feats.scp --nlsyms ${nlsyms} \
             data-stft/dt05_multi_noisy_ch${ch} ${dict} > data-stft/dt05_multi_noisy_ch${ch}/data.json
    done
fi


if [ -z ${tag} ]; then
    expdir=exp/${trainmode}_${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        echo "delta feature is not supported"
        exit
#        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

# noisy
train_feat_noisy="scp:data-stft/tr05_multi_noisy/feats.scp"
valid_feat_noisy="scp:data-stft/dt05_multi_isolated_1ch_track/feats.scp"

# enhan
train_feat_enhan=""
valid_feat_enhan=""
stft_ch="1 3 4 5 6"
for ch in ${stft_ch}; do
    train_feat_enhan="${train_feat_enhan} scp:data-stft/tr05_multi_noisy_ch${ch}/feats.scp"
    valid_feat_enhan="${valid_feat_enhan} scp:data-stft/dt05_multi_noisy_ch${ch}/feats.scp"
done

recog_set="\
dt05_real dt05_simu et05_real et05_simu
"

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
    ${cuda_cmd} ${expdir}/train.log \
        asr_mc_train.py \
        --gpu ${gpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-feat-noisy ${train_feat_noisy} \
        --valid-feat-noisy ${valid_feat_noisy} \
        --train-label-noisy data-stft/tr05_multi_noisy/data.json \
        --valid-label-noisy data-stft/dt05_multi_isolated_1ch_track/data.json \
        --train-feat-enhan ${train_feat_enhan} \
        --valid-feat-enhan ${valid_feat_enhan} \
        --train-label-enhan data-stft/tr05_multi_noisy_ch1/data.json \
        --valid-label-enhan data-stft/dt05_multi_noisy_ch1/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --ctc_type ${ctctype} \
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
        --melmat ${melmat} \
        --cmvn ${cmvn} \
        --mode ${trainmode}
fi

if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
#    nj=32
    nj=8

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${recogmode}_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}

        # noisy
        # split data
        data=data-stft/${rtask}_isolated_1ch_track
        split_data.sh --per-utt ${data} ${nj};
        # make json labels for recognition
        data2json.sh ${data} ${dict} > ${data}/data.json
        
        recog_feat_noisy="scp:data-stft/${rtask}_isolated_1ch_track/split${nj}utt/JOB/feats.scp"

        # enhan
        # split_data
        stft_ch="1 3 4 5 6"
        for ch in ${stft_ch}; do
            data=data-stft/${rtask}_noisy_ch${ch}
            split_data.sh --per-utt ${data} ${nj};
        done
        # make json labels for recognition
        data=data-stft/${rtask}_noisy_ch1
        data2json.sh ${data} ${dict} > ${data}/data.json

        recog_feat_enhan=""
        for ch in ${stft_ch}; do
            recog_feat_enhan="${recog_feat_enhan} scp:data-stft/${rtask}_noisy_ch${ch}/split${nj}utt/JOB/feats.scp"
        done

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_mc_recog.py \
            --gpu ${gpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-feat-noisy ${recog_feat_noisy} \
            --recog-label-noisy data-stft/${rtask}_isolated_1ch_track/data.json \
            --recog-feat-enhan ${recog_feat_enhan} \
            --recog-label-enhan data-stft/${rtask}_noisy_ch1/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --mode ${recogmode} \
            &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

