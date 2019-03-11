#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=1000000
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# configuration path
preprocess_conf=conf/preprocess.json

# network architecture
use_beamformer=true
blayers=3
bunits=300
bprojs=300

# encoder related
etype=vggblstmp     # encoder architecture type
elayers=3
eunits=1024
eprojs=1024
subsample=1_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=1024
# attention related
atype=location
adim=1024
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=10
maxlen_in=600  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
opt=adadelta
epochs=12
patience=0

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
lm_batchsize=300    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_patience=3
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=20
penalty=0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# enhanced speech option
fs=16000

# scheduled sampling option
samp_prob=0.0

# data
chime4_data=/export/corpora4/CHiME4/CHiME3 # JHU setup
wsj0=/export/corpora5/LDC/LDC93S6B            # JHU setup
wsj1=/export/corpora5/LDC/LDC94S13B           # JHU setup
chime4_data=/data/rigel1/corpora/CHiME4
wsj0=/data/rigel1/corpora/LDC93S6A
wsj1=/data/rigel1/corpora/LDC94S13A

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

which sox &> /dev/null || { echo "Error: requires sox command."; exit 1; }

train_set=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
train_dev=dt05_multi_isolated_6ch_track
recog_set="dt05_real_isolated_6ch_track dt05_simu_isolated_6ch_track et05_real_isolated_6ch_track et05_simu_isolated_6ch_track"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ## Task dependent. You have to make the following data preparation part by yourself.
    # But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    wsj0_data=${chime4_data}/data/WSJ0
    local/clean_wsj0_data_prep.sh ${wsj0_data}
    local/clean_chime4_format_data.sh
    echo "prepartion for chime4 data"
    local/real_noisy_chime4_data_prep.sh ${chime4_data}
    local/simu_noisy_chime4_data_prep.sh ${chime4_data}
    local/bth_chime4_data_prep.sh ${chime4_data}
    echo "test data for 6ch track"
    local/real_enhan_chime4_data_prep.sh isolated_6ch_track ${chime4_data}/data/audio/16kHz/isolated_6ch_track
    local/simu_enhan_chime4_data_prep.sh isolated_6ch_track ${chime4_data}/data/audio/16kHz/isolated_6ch_track

    # Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ## But you can utilize Kaldi recipes in most cases
    echo "stage 1: Dump wav files into a HDF5 file"

    echo "combine real and simulation data"
    # utils/combine_data.sh data/tr05_multi_noisy data/tr05_simu_noisy data/tr05_real_noisy
    for setname in tr05_multi_noisy ${recog_set};do
    # for setname in dt05_bth et05_bth;do
        mkdir -p data/${setname}_multich
        <data/${setname}/utt2spk sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u >data/${setname}_multich/utt2spk
        <data/${setname}/text sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u > data/${setname}_multich/text
        <data/${setname}_multich/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}_multich/spk2utt

        # 2th mic is omitted in default
        for ch in 1 3 4 5 6; do
            <data/${setname}/wav.scp grep "CH${ch}" | sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' >data/${setname}_multich/wav_ch${ch}.scp
        done
        mix-mono-wav-scp.py data/${setname}_multich/wav_ch*.scp > data/${setname}_multich/wav.scp
        rm -f data/${setname}_multich/wav_ch*.scp
    done

    # Note that data/tr05_multi_noisy_multich has multi-channel wav data, while data/train_si284 has 1ch only
    dump_pcm.sh --nj 32 --cmd ${train_cmd} --filetype "sound.hdf5" --format flac data/train_si284
    for setname in tr05_multi_noisy ${recog_set}; do
        dump_pcm.sh --nj 32 --cmd ${train_cmd} --filetype "sound.hdf5" --format flac data/${setname}_multich
    done
    utils/combine_data.sh data/${train_dev}_multich data/dt05_simu_isolated_6ch_track_multich data/dt05_real_isolated_6ch_track_multich

fi

train_set="${train_set}_multich"
train_dev="${train_dev}_multich"
# Rename recog_set: e.g. dt05_real_isolated_6ch_track -> dt05_real_isolated_6ch_track_multich
recog_set="$(for setname in ${recog_set}; do echo -n "${setname}_multich "; done)"


dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}${adim}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs$((batchsize * ngpu))_mli${maxlen_in}_mlo${maxlen_out}_usebf${use_beamformer}_bunit${bunits}_bproj${bprojs}
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    for setname in tr05_multi_noisy_multich ${train_dev} ${recog_set}; do
        data2json.sh --cmd "${train_cmd}" --nj 30 \
            --category "multichannel" \
            --preprocess-conf ${preprocess_conf} --filetype sound.hdf5 \
            --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
            --out data/${setname}/data.json data/${setname} ${dict}
    done

    setname=train_si284
    data2json.sh --cmd "${train_cmd}" --nj 30 \
        --category "singlechannel" \
        --preprocess-conf ${preprocess_conf} --filetype sound.hdf5 \
        --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
        --out data/${setname}/data.json data/${setname} ${dict}

    mkdir -p data/${train_set}
    concatjson.py data/tr05_multi_noisy_multich/data.json data/train_si284/data.json > data/${train_set}/data.json
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
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
        --dict ${lmdict}
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training: expdir=${expdir}"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --use-frontend True \
        --use-beamformer ${use_beamformer} \
        --blayers ${blayers} \
        --bunits ${bunits} \
        --bprojs ${bprojs} \
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
        --train-json data/${train_set}/data.json \
        --valid-json data/${train_dev}/data.json \
        --preprocess-conf ${preprocess_conf} \
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
        --sampling-probability ${samp_prob} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --sortagrad ${sortagrad} \
        --epochs ${epochs} \
        --patience ${patience}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=200

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}

        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi

        # split data
        splitjson.py --parts ${nj} data/${rtask}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lm-weight ${lm_weight} \
            ${recog_opts}
        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait

    echo "Decoding successfully finished"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Enhance speech"
    nj=200
    ngpu=0

    for rtask in ${recog_set}; do
    (
        enhdir=${expdir}/enhance_${rtask}
        mkdir -p ${enhdir}/outdir
        splitjson.py --parts ${nj} data/${rtask}/data.json

        ${decode_cmd} JOB=1:${nj} ${enhdir}/log/enhance.JOB.log \
            asr_enhance.py \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --debugmode ${debugmode} \
                --model ${expdir}/results/${recog_model}  \
                --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
                --enh-wspecifier ark,scp:${enhdir}/outdir/enhance.JOB,${enhdir}/outdir/enhance.JOB.scp \
                --enh-filetype "sound" \
                --image-dir ${enhdir}/images \
                --fs ${fs}
    ) &
    done
    wait

    for rtask in ${recog_set}; do
        enhdir=${expdir}/enhance_${rtask}
        for i in $(seq 1 ${nj}); do
            cat ${enhdir}/outdir/enhance.${i}.scp
        done > ${enhdir}/enhance.scp
    done
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Evaluate enhanced speech"
    for rtask in ${recog_set}; do
    (
        rtask=$(echo ${rtask} | sed -e 's/_multich//')

        for place in PED CAF STR BUS; do
            basedir=eval_mixture/${rtask}/eval_${place}
            bth=$(echo ${rtask} | sed -r "s/(dt05|et05).*/\1_bth/")
            mkdir -p ${basedir}

            <data/${bth}/wav.scp grep CH0 | sed -r "s/^[^_]*_(.*?)_BTH.CH[0-9] /\1 /g" | sort > ${basedir}/reference.scp
            <data/${rtask}/wav.scp grep CH5 | grep ${place} | sed -r "s/^[^_]*_([^_]*?)_${place}.CH5_[A-Z]... /\1 /" | sort > ${basedir}/target.scp
            eval_source_separation.sh --cmd ${train_cmd} --nj 100 --source-image false ${basedir}/reference.scp ${basedir}/target.scp ${basedir}

        done
    ) &
    done
    wait

    mkdir -p ${expdir}/eval_enhance
    for rtask in ${recog_set}; do
    (

        enhdir=${expdir}/enhance_${rtask}
        for place in PED CAF STR BUS; do
            basedir=${enhdir}/eval_${place}
            bth=$(echo ${rtask} | sed -r "s/(dt05|et05).*/\1_bth/")
            mkdir -p ${basedir}

            <data/${bth}/wav.scp grep CH0 | sed -r "s/^[^_]*_(.*?)_BTH.CH[0-9] /\1 /g" | sort > ${basedir}/reference.scp
            <${enhdir}/enhance.scp grep ${place} | sed -r "s/^[^_]*_(.*?)_${place}_(REAL|SIMU) /\1 /g" | sort > ${basedir}/target.scp
            eval_source_separation.sh --cmd ${train_cmd} --nj 100  --source-image false ${basedir}/reference.scp ${basedir}/target.scp ${basedir}
        done
    ) &
    done
    wait
fi

