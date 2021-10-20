#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

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

# data
chime4_data=/export/corpora4/CHiME4/CHiME3    # JHU setup
wsj0=/export/corpora5/LDC/LDC93S6B            # JHU setup
wsj1=/export/corpora5/LDC/LDC94S13B           # JHU setup

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

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
    echo "preparation for chime4 data"
    local/real_noisy_chime4_data_prep.sh ${chime4_data}
    local/simu_noisy_chime4_data_prep.sh ${chime4_data}
    local/bth_chime4_data_prep.sh ${chime4_data}
    echo "test data for 6ch track"
    local/real_enhan_chime4_data_prep.sh isolated_6ch_track ${chime4_data}/data/audio/16kHz/isolated_6ch_track
    local/simu_enhan_chime4_data_prep.sh isolated_6ch_track ${chime4_data}/data/audio/16kHz/isolated_6ch_track

    # Fix text because local/*_noisy_chime4_data_prep.sh assumes 1ch_track originally.
    for setname in dt05_real_isolated_6ch_track dt05_simu_isolated_6ch_track et05_real_isolated_6ch_track et05_simu_isolated_6ch_track; do
        for ch in 0 1 2 3 4 5 6; do
            # Skip if ch == 1 and simulation
            [ ${ch} -eq 0 ] && echo ${setname} | grep simu &> /dev/null && continue

            # e.g. F05_440C0202_BUS_SIMU -> F05_440C0202_BUS.CH1_SIMU
             <data/${setname}/text sed -r "s/_([A-Z]*)_([A-Z]*) /_\1.CH${ch}_\2 /"
        done | sort > data/${setname}/text.tmp
        mv data/${setname}/text.tmp data/${setname}/text
        ./utils/validate_data_dir.sh --no-feats data/${setname}
    done

    for setname in dt05_bth et05_bth; do
        ./utils/validate_data_dir.sh --no-feats data/${setname}
    done

    # Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ## But you can utilize Kaldi recipes in most cases
    echo "stage 1: Dump wav files into a HDF5 file"

    utils/combine_data.sh data/tr05_multi_noisy data/tr05_simu_noisy data/tr05_real_noisy
    for setname in tr05_multi_noisy ${recog_set}; do
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
    dump_pcm.sh --nj 32 --cmd "${train_cmd}" --filetype "sound.hdf5" --format flac data/train_si284
    for setname in tr05_multi_noisy ${recog_set}; do
        dump_pcm.sh --nj 32 --cmd "${train_cmd}" --filetype "sound.hdf5" --format flac data/${setname}_multich
    done
    utils/combine_data.sh data/${train_set}_multich data/tr05_multi_noisy_multich data/train_si284
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
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
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

    # "--category" is just a mark to be used excludely for minibatch creation.
    # i.e. A minibatch always consists of the samples in either one of these categories.

    echo "make json files"
    for setname in tr05_multi_noisy_multich ${train_dev} ${recog_set}; do
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
    concatjson.py data/tr05_multi_noisy_multich/data.json data/train_si284/data.json > data/${train_set}/data.json
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
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}

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
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Evaluate enhanced speech"
    nj=32

    mkdir -p ${expdir}/eval_enhance
    pids=() # initialize pids
    for rtask in ${recog_set}; do
        # SKip real data because there are no clean signal
        echo ${rtask} | grep real &> /dev/null && continue
    (

        enhdir=${expdir}/enhance_${rtask}
        for place in PED CAF STR BUS; do
            basedir=${enhdir}/eval_${place}
            bth=$(echo ${rtask} | sed -r "s/(dt05|et05).*/\1_bth/")
            mkdir -p ${basedir}

            # CH0 is worn mic
            <data/${bth}/wav.scp grep CH0 | sed -r "s/^[^_]*_(.*?)_BTH.CH[0-9] /\1 /g" | sort > ${basedir}/reference.scp
            <${enhdir}/enhance.scp grep ${place} | sed -r "s/^[^_]*_(.*?)_${place}_(REAL|SIMU) /\1 /g" | sort > ${basedir}/estimated.scp
            # FIME(kamo): Should we use bss_eval_images?
            eval_source_separation.sh --cmd "${decode_cmd}" --nj ${nj} --bss-eval-images false ${basedir}/reference.scp ${basedir}/estimated.scp ${basedir}
        done
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    local/show_enhance_results.sh ${expdir}/enhance_

    # Computes SDR/STOI/PESQ, etc., between noisy and clean signals.
    # The resutl can be seen in ./RESULT, so you don't need to run this block.
    if false; then
        pids=() # initialize pids
        for rtask in ${recog_set}; do
            # SKip real data because there are no clean signal
            echo ${rtask} | grep real &> /dev/null && continue
        (
            rtask=${rtask/_multich//}

            for place in PED CAF STR BUS; do
                basedir=eval_noisy/${rtask}/eval_${place}
                bth=$(echo ${rtask} | sed -r "s/(dt05|et05).*/\1_bth/")
                mkdir -p ${basedir}

                # CH0 is worn mic
                <data/${bth}/wav.scp grep CH0 | sed -r "s/^[^_]*_(.*?)_BTH.CH[0-9] /\1 /g" | sort > ${basedir}/reference.scp
                # Use CH5 as reference
                <data/${rtask}/wav.scp grep CH5 | grep ${place} | sed -r "s/^[^_]*_([^_]*?)_${place}.CH5_[A-Z]... /\1 /" | sort > ${basedir}/estimated.scp

                eval_source_separation.sh --cmd "${decode_cmd}" --nj ${nj} --bss-eval-images false ${basedir}/reference.scp ${basedir}/estimated.scp ${basedir}

            done
        ) &
        pids+=($!) # store background pids
        done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

        local/show_enhance_results.sh eval_noisy/
    fi
fi
