#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0         # start from -1 if you need to start from data download
stop_stage=5
ngpu=1          # number of gpus during training ("0" uses cpu, otherwise use gpu)
dec_ngpu=0      # number of gpus during decoding ("0" uses cpu, otherwise use gpu)
nj=8            # number of parallel jobs for decoding
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
decode_config=conf/decode.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.
metric=bleu                  # loss/acc/bleu

# pre-training related
asr_model=
mt_model=

# preprocessing related
src_case=lc.rm
tgt_case=tc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# postprocessing related
remove_nonverbal=true  # remove non-verbal labels such as "( Applaus )"
# NOTE: IWSLT community accepts this setting and therefore we use this by default

# segmentation related
max_interval=100
max_duration=2000

# bpemode (unigram or bpe)
nbpe=16000
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

# test data directory
iwslt_test_data_dir=/n/rd8/iwslt18

train_set=train.de
train_dev=dev.de
trans_subset="et_mustc_dev_org.de et_mustc_tst-COMMON.de et_mustc_tst-HE.de"
trans_set="et_mustc_dev_org.de et_mustc_tst-COMMON.de et_mustc_tst-HE.de \
           et_mustcv2_dev_org.de et_mustcv2_tst-COMMON.de et_mustcv2_tst-HE.de"
iwslt_test_set="et_stted_dev2010.de et_stted_tst2010.de et_stted_tst2013.de et_stted_tst2014.de et_stted_tst2015.de \
                et_stted_tst2018.de et_stted_tst2019.de"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"

    # Must-C
    if [ ! -d "${mustc_dir}/st1/data/train_sp.en-de.en" ]; then
        echo "run ${mustc_dir}/st1/run.sh first"
        exit 1
    fi
    data_code=mustc
    for lang in en de; do
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/train_sp.en-de.${lang}   data/tr_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/dev.en-de.${lang}        data/dt_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/dev_org.en-de.${lang}    data/et_${data_code}_dev_org.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/tst-COMMON.en-de.${lang} data/et_${data_code}_tst-COMMON.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/tst-HE.en-de.${lang}     data/et_${data_code}_tst-HE.${lang}
    done

    # Must-C v2
    if [ ! -d "${mustc_v2_dir}/st1/data/train_sp.en-de.en" ]; then
        echo "run ${mustc_v2_dir}/st1/run.sh first"
        exit 1
    fi
    data_code=mustcv2
    for lang in en de; do
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/st1/data/train_sp.en-de.${lang}   data/tr_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/st1/data/dev.en-de.${lang}        data/dt_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/st1/data/dev_org.en-de.${lang}    data/et_${data_code}_dev_org.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/st1/data/tst-COMMON.en-de.${lang} data/et_${data_code}_tst-COMMON.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_v2_dir}/st1/data/tst-HE.en-de.${lang}     data/et_${data_code}_tst-HE.${lang}
    done

    # ST-TED
    if [ ! -d "${stted_dir}/st1/data/train_nodevtest_sp.en" ]; then
        echo "run ${stted_dir}/st1/run.sh first"
        exit 1
    fi
    data_code=stted
    for lang in en de; do
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/st1/data/train_nodevtest_sp.${lang} data/tr_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/st1/data/train_dev.${lang}          data/dt_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/st1/data/dev.${lang}                data/et_${data_code}_dev.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${stted_dir}/st1/data/test.${lang}               data/et_${data_code}_test.${lang}
    done
    # En-De only
    for x in dev2010 tst2010 tst2010 tst2013 tst2014 tst2015 tst2018 tst2019; do
        cp -rf ${stted_dir}/st1/data/${x}.en data/et_${data_code}_${x}.en
        cp -rf ${stted_dir}/st1/data/${x}.de data/et_${data_code}_${x}.de
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
    for x in tr_mustc.de tr_mustcv2.de tr_stted.de dt_mustc.de dt_mustcv2.de dt_stted.de ${trans_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${x}
        rm data/${x}/segments
        rm data/${x}/wav.scp
    done

    for lang in en de; do
        utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/train.${lang} data/tr_mustc.${lang} data/tr_mustcv2.${lang} data/tr_stted.${lang}
        utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/dev.${lang}   data/dt_mustc.${lang} data/dt_mustcv2.${lang} data/dt_stted.${lang}
    done

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
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    for x in ${train_dev} ${trans_set}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}; mkdir -p ${feat_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${x}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/trans/${x} ${feat_dir}
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

        feat_dir=${dumpdir}/${output_dir}/delta${do_delta}; mkdir -p ${feat_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${output_dir}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/trans/${output_dir} ${feat_dir}
    done
fi

dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${tgt_case}.txt
nlsyms=data/lang_1spm/${train_set}_non_lang_syms_${tgt_case}.txt
bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${tgt_case}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1spm/

    echo "make a non-linguistic symbol list for all languages"
    grep sp1.0 data/train.*/text.${tgt_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a joint source and target dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=$(wc -l < ${dict})
    grep sp1.0 data/train.*/text.${tgt_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_1spm/input_${src_case}_${tgt_case}.txt
    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input_${src_case}_${tgt_case}.txt \
        --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input_${src_case}_${tgt_case}.txt \
        | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "de" \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
    for x in ${train_dev} ${trans_set} ${iwslt_test_set}; do
        if [[ ${x} = *tst20* ]] || [[ ${x} = *dev20* ]]; then
            feat_dir=${dumpdir}/${x}_merge${max_interval}_duration${max_duration}/delta${do_delta}
            local/data2json.sh --feat ${feat_dir}/feats.scp --no_text true \
                data/${x}_merge${max_interval}_duration${max_duration} ${dict} > ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        else
            feat_dir=${dumpdir}/${x}/delta${do_delta}
            data2json.sh --feat ${feat_dir}/feats.scp --text data/${x}/text.${tgt_case} --bpecode ${bpemodel}.model --lang "de" \
                data/${x} ${dict} > ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json
        fi
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev} ${trans_set}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/$(echo ${x} | cut -f 1 -d ".").en
        update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
            ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json ${data_dir} ${dict}
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${tgt_case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
else
    expname=${train_set}_${tgt_case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        st_train.py \
        --config ${train_config} \
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
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
        --enc-init ${asr_model} \
        --dec-init ${mt_model} \
        --n-iter-processes 3
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        # Average ST models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log --metric ${metric}"
        else
            trans_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${trans_model} \
            --num ${n_average}
    fi

    if [ ${dec_ngpu} = 1 ]; then
        nj=1
    fi

    pids=() # initialize pids
    for x in ${trans_subset}; do
    (
        if [[ ${x} = *tst20* ]] || [[ ${x} = *dev20* ]]; then
            x=${x}_merge${max_interval}_duration${max_duration}
        fi
        decode_dir=decode_${x}_$(basename ${decode_config%.*})
        feat_dir=${dumpdir}/${x}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            st_trans.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        if [[ ${x} = *tst20* ]] || [[ ${x} = *dev20* ]]; then
            set=$(echo ${x} | cut -f 1 -d "." | cut -f 3 -d "_")
            local/score_bleu_reseg.sh --case ${tgt_case} --bpemodel ${bpemodel}.model \
                --remove_nonverbal ${remove_nonverbal} \
                ${expdir}/${decode_dir} ${dict} ${iwslt_test_data_dir} ${set}
        else
            score_bleu.sh --case ${tgt_case} --bpemodel ${bpemodel}.model \
                --remove_nonverbal ${remove_nonverbal} \
                ${expdir}/${decode_dir} "de" ${dict}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
