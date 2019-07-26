#!/bin/bash

# Copyright 2018 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# configuration path
preprocess_config=conf/preprocess.json
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# enhanced speech option
fs=16000

# Dereverberation Measures
compute_se=true # flag for turing on computation of dereverberation measures
enable_pesq=true # please make sure that you or your institution have the license to report PESQ before turning on this flag
nch_se=8

# data
reverb=/export/corpora5/REVERB_2014/REVERB    # JHU setup
wsjcam0=/export/corpora3/LDC/LDC95S24/wsjcam0 # JHU setup
wsj0=/export/corpora5/LDC/LDC93S6B            # JHU setup
wsj1=/export/corpora5/LDC/LDC94S13B           # JHU setup
wavdir=${PWD}/wav # place to store WAV files

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

train_set=tr_simu_2ch_si284
train_dev=dt_multi_2ch
recog_set="dt_simu_8ch dt_real_8ch et_simu_8ch et_real_8ch"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make the following data preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    wavdir=${PWD}/wav # set the directory of the multi-condition training WAV files to be generated
    echo "stage 0: Data preparation"
    local/generate_data.sh --wavdir ${wavdir} ${wsjcam0}
    local/prepare_simu_data.sh --wavdir ${wavdir} ${reverb} ${wsjcam0}
    local/prepare_real_data.sh --wavdir ${wavdir} ${reverb}

    # Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Dump wav files into a HDF5 file"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    tasks="${recog_set} tr_simu_2ch"
    tasks_2ch="tr_simu_2ch dt_simu_2ch dt_real_2ch"
    for setname in ${recog_set}; do
        echo ${setname}
        mkdir -p data/${setname}_multich
        <data/${setname}/utt2spk sed -r 's/^(.*?)_[A-H](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/utt2spk
        <data/${setname}/text sed -r 's/^(.*?)_[A-H](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/text
        <data/${setname}_multich/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}_multich/spk2utt

        for ch in {A..H}; do
            <data/${setname}/wav.scp grep "_${ch}_" | sed -r 's/^(.*?)_[A-H](_.*?) /\1\2 /g' >data/${setname}_multich/wav_ch${ch}.scp
        done
        mix-mono-wav-scp.py data/${setname}_multich/wav_ch*.scp > data/${setname}_multich/wav.scp
    done

    for setname in ${tasks_2ch}; do
        echo ${setname}
        mkdir -p data/${setname}_multich
        <data/${setname}/utt2spk sed -r 's/^(.*?)_[A-B](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/utt2spk
        <data/${setname}/text sed -r 's/^(.*?)_[A-B](_.*?) /\1\2 /g' | sort -u > data/${setname}_multich/text
        <data/${setname}_multich/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}_multich/spk2utt

        for ch in {A..B}; do
            <data/${setname}/wav.scp grep "_${ch}_" | sed -r 's/^(.*?)_[A-B](_.*?) /\1\2 /g' >data/${setname}_multich/wav_ch${ch}.scp
        done
        mix-mono-wav-scp.py data/${setname}_multich/wav_ch*.scp > data/${setname}_multich/wav.scp
    done

    # Note that data/tr05_multi_noisy_multich has multi-channel wav data, while data/train_si284 has 1ch only
    dump_pcm.sh --nj 32 --cmd "${train_cmd}" --filetype "sound.hdf5" data/train_si284
    for setname in ${tasks_2ch} ${recog_set}; do
        dump_pcm.sh --nj 32 --cmd "${train_cmd}" --filetype "sound.hdf5" data/${setname}_multich
    done
    echo "combine real and simulation development data"
    utils/combine_data.sh data/${train_dev}_multich data/dt_real_2ch_multich data/dt_simu_2ch_multich
    echo "combine reverb simulation and wsj clean training data"
    utils/combine_data.sh data/${train_set}_multich data/train_si284 data/tr_simu_2ch
fi

train_set="${train_set}_multich"
train_dev="${train_dev}_multich"
# Rename recog_set: e.g. dt05_real_isolated_6ch_track -> dt05_real_isolated_6ch_track_multich
recog_set="$(for setname in ${recog_set}; do echo -n "${setname}_multich "; done)"

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    utils/combine_data.sh data/${train_set} data/train_si284 data/tr_simu_2ch
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
    for setname in tr_simu_2ch_multich ${train_dev} ${recog_set}; do
        data2json.sh --cmd "${train_cmd}" --nj 30 \
        --category "multichannel" \
        --preprocess-conf ${preprocess_config} --filetype sound.hdf5 \
        --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
        --out data/${setname}/data.json data/${setname} ${dict}
    done

    setname=train_si284
    data2json.sh --cmd "${train_cmd}" --nj 30 \
    --category "singlechannel" \
    --preprocess-conf ${preprocess_config} --filetype sound.hdf5 \
    --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
    --out data/${setname}/data.json data/${setname} ${dict}

    mkdir -p data/${train_set}
    concatjson.py data/tr_simu_2ch_multich/data.json data/train_si284/data.json > data/${train_set}/data.json
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
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
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training: expdir=${expdir}"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
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
        --preprocess-conf ${preprocess_config}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        if [ ${use_wordlm} = true ]; then
            decode_dir=${decode_dir}_wordrnnlm_${lmtag}
        else
            decode_dir=${decode_dir}_rnnlm_${lmtag}
        fi
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
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Report the results"
    decode_part_dir=$(basename ${decode_config%.*})
    if [ ${use_wordlm} = true ]; then
	decode_part_dir=${decode_part_dir}_wordrnnlm_${lmtag}
    else
	decode_part_dir=${decode_part_dir}_rnnlm_${lmtag}
    fi
    local/get_results.sh ${nlsyms} ${dict} ${expdir} ${decode_part_dir}
    echo "Decoding successfully finished"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Enhance speech"
    nj=32
    ngpu=0

    pids=() # initialize pids
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
                --num-images 20 \
                --fs ${fs}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    # Reduce all scp files from each jobs to one
    for rtask in ${recog_set}; do
        enhdir=${expdir}/enhance_${rtask}
        for i in $(seq 1 ${nj}); do
            cat ${enhdir}/outdir/enhance.${i}.scp
        done > ${enhdir}/enhance.scp
    done
    echo "Enhancement successfully finished"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Score enhanced speech"
    if ${compute_se}; then
        if [ ! -d local/REVERB_scores_source ] || [ ! -d local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/SRMRToolbox ] || [ ! -f local/PESQ ]; then
            # download and install speech enhancement evaluation tools
            local/download_se_eval_tool.sh
        fi
        pesqdir=${PWD}/local
        for rtask in "et"; do
            simu_scp=${expdir}/enhance_${rtask}_simu_${nch_se}ch_multich/enhance.scp
            real_scp=${expdir}/enhance_${rtask}_real_${nch_se}ch_multich/enhance.scp
            enhancement_simu_scp=${PWD}/$simu_scp
            enhancement_real_scp=${PWD}/$real_scp
            local/compute_se_scores.sh --nch ${nch_se} --enable_pesq ${enable_pesq} \
                ${enhancement_simu_scp} \
                ${enhancement_real_scp} \
                ${reverb} \
                ${PWD}/data/${rtask}_cln/wav.scp \
                ${pesqdir} \
                ${expdir}/enhanced_${rtask}_${nch_se}ch_metrics

            cat ${expdir}/enhanced_${rtask}_metrics/scores/score_SimData
            cat ${expdir}/enhanced_${rtask}_metrics/scores/score_RealData
        done
    fi
fi
