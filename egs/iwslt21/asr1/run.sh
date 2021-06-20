#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=0         # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus during training ("0" uses cpu, otherwise use gpu)
dec_ngpu=0      # number of gpus during decoding ("0" uses cpu, otherwise use gpu)
nj=32           # number of parallel jobs for decoding
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.

# iwslt segmentation related
max_interval=200
max_duration=2000

# bpemode (unigram or bpe)
nbpe=5000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# data directories
mustc_dir=../../must_c
mustc_v2_dir=../../must_c_v2
stted_dir=../../iwslt18
tedlium2_dir=../../tedlium2
librispeech_dir=../../librispeech

# test data directory
iwslt_test_data_dir=/n/rd8/iwslt18

train_set=train
train_dev=dev
recog_subset="et_mustc_tst-COMMON et_tedlium2_test et_librispeech_test_other"  # for quick decoding
recog_set="et_mustc_dev_org et_mustc_tst-COMMON et_mustc_tst-HE \
           et_mustcv2_dev_org et_mustcv2_tst-COMMON et_mustcv2_tst-HE \
           et_stted_dev et_stted_test \
           et_tedlium2_dev et_tedlium2_test \
           et_librispeech_dev_clean et_librispeech_dev_other et_librispeech_test_clean et_librispeech_test_other"
iwslt_test_set="et_stted_dev2010 et_stted_tst2010 et_stted_tst2013 et_stted_tst2014 et_stted_tst2015 \
                et_stted_tst2018 et_stted_tst2019"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"

    # Must-C
    if [ ! -d "${mustc_dir}/asr1/data/train_sp.en-de.en" ]; then
        echo "run ${mustc_dir}/asr1/run.sh first"
        exit 1
    fi
    data_code=mustc
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/asr1/data/train_sp.en-de.en   data/tr_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/asr1/data/dev.en-de.en        data/dt_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/asr1/data/dev_org.en-de.en    data/et_${data_code}_dev_org
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/asr1/data/tst-COMMON.en-de.en data/et_${data_code}_tst-COMMON
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/asr1/data/tst-HE.en-de.en     data/et_${data_code}_tst-HE

    # Must-C v2
    if [ ! -d "${mustc_v2_dir}/asr1/data/train_sp.en-de.en" ]; then
        echo "run ${mustc_v2_dir}/asr1/run.sh first"
        exit 1
    fi
    data_code=mustcv2
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/asr1/data/train_sp.en-de.en   data/tr_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/asr1/data/dev.en-de.en        data/dt_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/asr1/data/dev_org.en-de.en    data/et_${data_code}_dev_org
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/asr1/data/tst-COMMON.en-de.en data/et_${data_code}_tst-COMMON
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/asr1/data/tst-HE.en-de.en     data/et_${data_code}_tst-HE

    # ST-TED
    if [ ! -d "${stted_dir}/asr1/data/train_nodevtest_sp.en" ]; then
        echo "run ${stted_dir}/asr1/run.sh first"
        exit 1
    fi
    data_code=stted
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/asr1/data/train_nodevtest_sp.en data/tr_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/asr1/data/train_dev.en          data/dt_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/asr1/data/dev.en                data/et_${data_code}_dev
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/asr1/data/test.en               data/et_${data_code}_test
    for x in dev2010 tst2010 tst2010 tst2013 tst2014 tst2015 tst2018 tst2019; do
        cp -rf ${stted_dir}/asr1/data/${x}.en data/et_${data_code}_${x}
    done

    # TEDLIUM2
    if [ ! -d "${tedlium2_dir}/asr1/data/train_trim_sp" ]; then
        echo "run ${tedlium2_dir}/asr1/run.sh first"
        exit 1
    fi
    data_code=tedlium2
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${tedlium2_dir}/asr1/data/train_trim_sp data/tr_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${tedlium2_dir}/asr1/data/dev_trim      data/dt_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${tedlium2_dir}/asr1/data/dev           data/et_${data_code}_dev
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${tedlium2_dir}/asr1/data/test          data/et_${data_code}_test
    # additionally we copy text to text.${case}
    for x in tr_${data_code} dt_${data_code} et_${data_code}_dev et_${data_code}_test; do
        for case in tc lc lc.rm; do
            paste -d " " <(cut -d " " -f 1 data/${x}/text) <(cut -d " " -f 2- data/${x}/text | lowercase.perl | tokenizer.perl -l en -q) > data/${x}/text.${case}
        done
    done

    # Librispeech
    if [ ! -d "${librispeech_dir}/asr1/data/train_sp" ]; then
        echo "run ${librispeech_dir}/asr1/run.sh first"
        exit 1
    fi
    data_code=librispeech
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${librispeech_dir}/asr1/data/train_960  data/tr_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${librispeech_dir}/asr1/data/dev        data/dt_${data_code}
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${librispeech_dir}/asr1/data/dev_clean  data/et_${data_code}_dev_clean
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${librispeech_dir}/asr1/data/dev_other  data/et_${data_code}_dev_other
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${librispeech_dir}/asr1/data/test_clean data/et_${data_code}_test_clean
    local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${librispeech_dir}/asr1/data/test_other data/et_${data_code}_test_other
    # additionally we copy text to text.${case}
    for x in tr_${data_code} dt_${data_code} et_${data_code}_dev_clean et_${data_code}_dev_other et_${data_code}_test_clean et_${data_code}_test_other; do
        # lowercase (all characters in Librispeech is truecased) + tokenization
        for case in tc lc lc.rm; do
            paste -d " " <(cut -d " " -f 1 data/${x}/text) <(cut -d " " -f 2- data/${x}/text | lowercase.perl | tokenizer.perl -l en -q) > data/${x}/text.${case}
        done
        cp data/${x}/text.lc.rm data/${x}/text
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in tr_mustc tr_mustcv2 tr_stted tr_tedlium2 tr_librispeech dt_mustc dt_mustcv2 dt_stted dt_tedlium2 dt_librispeech ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${x}
        rm data/${x}/segments
        rm data/${x}/wav.scp
    done

    utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/${train_set} data/tr_mustc data/tr_mustcv2 data/tr_librispeech data/tr_stted data/tr_tedlium2
    utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/${train_dev} data/dt_mustc data/dt_mustcv2 data/dt_librispeech data/dt_stted data/dt_tedlium2

    echo "Remove offlimit"
    cp -rf data/${train_set} data/${train_set}.tmp
    cp data/${train_set}/utt2spk data/${train_set}/utt2spk.org
    local/filter_offlimit.py --offlimit_list local/offlimit_list --utt2spk data/${train_set}/utt2spk.org > data/${train_set}/utt2spk
    utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${train_set}
    rm -rf data/${train_set}.tmp
    # NOTE: 5 speakers are expected to be removed

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt21/asr1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt21/asr1/dump/${train_dev}/delta${do_delta}/storage \
          ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    for x in ${train_dev} ${recog_set}; do
        feat_recog_dir=${dumpdir}/${x}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${x}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${x} ${feat_recog_dir}
    done

    # concatenate short segments
    for x in ${iwslt_test_set}; do
        output_dir=${x}_merge${max_interval}_duration${max_duration}
        rm -rf data/${output_dir}
        cp -rf data/${x} data/${output_dir}
        rm data/${output_dir}/utt2num_frames

        local/merge_short_segments.py \
            data/${x}/segments \
            data/${output_dir}/segments \
            data/${output_dir}/utt2spk \
            data/${output_dir}/spk2utt \
            --min_interval ${max_interval} \
            --max_duration ${max_duration} \
            --delimiter "_" || exit 1;

        # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${output_dir} exp/make_fbank/${output_dir} ${fbankdir}
        utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${output_dir}

        feat_recog_dir=${dumpdir}/${output_dir}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${output_dir}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/trans/${output_dir} ${feat_recog_dir}
    done
fi

dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units.txt
nlsyms=data/lang_1spm/non_lang_syms.txt
bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1spm/

    echo "make a non-linguistic symbol list for all languages"
    grep sp1.0 data/${train_set}/text.lc.rm | cut -f 2- -d' ' > data/lang_1spm/input.txt
    grep -o -P '&[^;]*;' data/lang_1spm/input.txt | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=$(wc -l < ${dict})
    grep -v -e '^\s*$' data/lang_1spm/input.txt > data/lang_1spm/input.txt.tmp
    mv data/lang_1spm/input.txt.tmp data/lang_1spm/input.txt
    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input.txt \
        --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input.txt \
        | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    wc -l ${dict}
    # NOTE: ASR vocab is created with a source language only

    echo "make json files"
    data2json.sh --nj 32 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text.lc.rm --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    for x in ${train_dev} ${recog_set} ${iwslt_test_set}; do
        if [[ ${x} = *tst20* ]] || [[ ${x} = *dev20* ]]; then
            feat_recog_dir=${dumpdir}/${x}_merge${max_interval}_duration${max_duration}/delta${do_delta}
            local/data2json.sh --feat ${feat_recog_dir}/feats.scp --no_text true \
                data/${x}_merge${max_interval}_duration${max_duration} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
        else
            feat_recog_dir=${dumpdir}/${x}/delta${do_delta}
            data2json.sh --feat ${feat_recog_dir}/feats.scp --text data/${x}/text.lc.rm --bpecode ${bpemodel}.model \
                data/${x} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
        fi
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 3)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=${train_set}_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_${train_set}_${bpemode}${nbpe}
    mkdir -p ${lmdatadir}
    # TEDLIUM2 external text
    gunzip -c ${tedlium2_dir}/asr1/db/TEDLIUM_release2/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    # LibriSpeech external text
    if [ ! -e ${librispeech_dir}/data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
        wget https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P ${librispeech_dir}/data/local/lm_train/
    fi
    zcat ${librispeech_dir}/data/local/lm_train/librispeech-lm-norm.txt.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece >> ${lmdatadir}/train.txt
    # transcription
    grep sp1.0 data/${train_set}/text.lc.rm | cut -f 2- -d " " | spm_encode --model=${bpemodel}.model --output_format=piece \
        >> ${lmdatadir}/train.txt
    cut -f 2- -d " " data/${train_dev}/text.lc.rm | spm_encode --model=${bpemodel}.model --output_format=piece \
        > ${lmdatadir}/valid.txt
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
        --dict ${dict} \
        --dump-hdf5-path ${lmdatadir}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --n-iter-processes 3 \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
       [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
       [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}
    fi

    if [ ${dec_ngpu} = 1 ]; then
        nj=1
    fi

    pids=() # initialize pids
    for x in ${recog_subset}; do
    (
        if [[ ${x} = *tst20* ]] || [[ ${x} = *dev20* ]]; then
            x=${x}_merge${max_interval}_duration${max_duration}
        fi
        decode_dir=decode_${x}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${x}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        if [[ ${x} = *tst20* ]] || [[ ${x} = *dev20* ]]; then
            set=$(echo ${x} | cut -f 3 -d "_")
            local/score_sclite_reseg.sh --case lc.rm --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true \
                ${expdir}/${decode_dir} ${dict} ${iwslt_test_data_dir} ${set}
        else
            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
