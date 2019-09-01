#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.

# pre-training related
asr_model=
mt_model=

# preprocessing related
case=lc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
st_ted=/export/b08/inaguma/IWSLT
# st_ted=/n/rd11/corpora_8/iwslt18

# bpemode (unigram or bpe)
nbpe=106
bpemode=bpe
# NOTE: nbpe=88 means character-level ST (lc.rm)
# NOTE: nbpe=106 means character-level ST (lc)
# NOTE: nbpe=134 means character-level ST (tc)

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodevtest_sp.de
train_set_prefix=train_nodevtest_sp
train_dev=train_dev.de
recog_set="dev.de test.de dev2010.de tst2010.de tst2013.de tst2014.de tst2015.de"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    for part in train dev2010 tst2010 tst2013 tst2014 tst2015; do
        local/download_and_untar.sh ${st_ted} ${part}
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    local/data_prep_train.sh ${st_ted}

    for part in dev2010 tst2010 tst2013 tst2014 tst2015; do
        local/data_prep_eval.sh ${st_ted} ${part}
    done

    # data cleaning
    ### local/forced_align.sh ${st_ted} data/train
    cp -rf data/train data/train.tmp
    reduce_data_dir.sh data/train.tmp data/local/downloads/reclist data/train
    for lang in en de; do
        utils/filter_scp.pl data/train/utt2spk <data/train.tmp/text.tc.${lang} >data/train/text.tc.${lang}
        utils/filter_scp.pl data/train/utt2spk <data/train.tmp/text.lc.${lang} >data/train/text.lc.${lang}
        utils/filter_scp.pl data/train/utt2spk <data/train.tmp/text.lc.rm.${lang} >data/train/text.lc.rm.${lang}
    done
    rm -rf data/train.tmp
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train dev2010 tst2010 tst2013 tst2014 tst2015; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # make a dev set
    utils/subset_data_dir.sh --speakers data/train 2000 data/dev
    utils/subset_data_dir.sh --spk-list <(utils/filter_scp.pl --exclude data/dev/spk2utt data/train/spk2utt) data/train data/train_nodev
    for lang in en de; do
        utils/filter_scp.pl data/train_nodev/utt2spk <data/train/text.tc.${lang} >data/train_nodev/text.tc.${lang}
        utils/filter_scp.pl data/train_nodev/utt2spk <data/train/text.lc.${lang} >data/train_nodev/text.lc.${lang}
        utils/filter_scp.pl data/train_nodev/utt2spk <data/train/text.lc.rm.${lang} >data/train_nodev/text.lc.rm.${lang}
        utils/filter_scp.pl data/dev/utt2spk <data/train/text.tc.${lang} >data/dev/text.tc.${lang}
        utils/filter_scp.pl data/dev/utt2spk <data/train/text.lc.${lang} >data/dev/text.lc.${lang}
        utils/filter_scp.pl data/dev/utt2spk <data/train/text.lc.rm.${lang} >data/dev/text.lc.rm.${lang}
    done

    # make a speaker-disjoint test set
    utils/subset_data_dir.sh --speakers data/train_nodev 2000 data/test
    utils/subset_data_dir.sh --spk-list <(utils/filter_scp.pl --exclude data/test/spk2utt data/train_nodev/spk2utt) data/train_nodev data/train_nodevtest
    for lang in en de; do
        utils/filter_scp.pl data/train_nodevtest/utt2spk <data/train_nodev/text.tc.${lang} >data/train_nodevtest/text.tc.${lang}
        utils/filter_scp.pl data/train_nodevtest/utt2spk <data/train_nodev/text.lc.${lang} >data/train_nodevtest/text.lc.${lang}
        utils/filter_scp.pl data/train_nodevtest/utt2spk <data/train_nodev/text.lc.rm.${lang} >data/train_nodevtest/text.lc.rm.${lang}
        utils/filter_scp.pl data/test/utt2spk <data/train_nodev/text.tc.${lang} >data/test/text.tc.${lang}
        utils/filter_scp.pl data/test/utt2spk <data/train_nodev/text.lc.${lang} >data/test/text.lc.${lang}
        utils/filter_scp.pl data/test/utt2spk <data/train_nodev/text.lc.rm.${lang} >data/test/text.lc.rm.${lang}
    done

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/train_nodevtest data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/train_nodevtest data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/train_nodevtest data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/train_nodevtest_sp data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        data/train_nodevtest_sp exp/make_fbank/train_nodevtest_sp ${fbankdir}
    for lang in en de; do
        awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' data/train_nodevtest/utt2spk > data/train_nodevtest_sp/utt_map
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.tc.${lang} >data/train_nodevtest_sp/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.lc.${lang} >data/train_nodevtest_sp/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.lc.rm.${lang} >data/train_nodevtest_sp/text.lc.rm.${lang}
        awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' data/train_nodevtest/utt2spk > data/train_nodevtest_sp/utt_map
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.tc.${lang} >>data/train_nodevtest_sp/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.lc.${lang} >>data/train_nodevtest_sp/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.lc.rm.${lang} >>data/train_nodevtest_sp/text.lc.rm.${lang}
        awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' data/train_nodevtest/utt2spk > data/train_nodevtest_sp/utt_map
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.tc.${lang} >>data/train_nodevtest_sp/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.lc.${lang} >>data/train_nodevtest_sp/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_nodevtest_sp/utt_map <data/train_nodevtest/text.lc.rm.${lang} >>data/train_nodevtest_sp/text.lc.rm.${lang}
    done

    # Divide into source and target languages
    for x in ${train_set_prefix} dev test dev2010 tst2010 tst2013 tst2014 tst2015; do
        local/divide_lang.sh ${x}
    done

    for lang in en de; do
        if [ -d data/train_dev.${lang} ];then
            rm -rf data/train_dev.${lang}
        fi
        cp -rf data/dev.${lang} data/train_dev.${lang}
    done

    for x in ${train_set_prefix} train_dev; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        for lang in en de; do
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
        done

        # Match the number of utterances between source and target languages
        # extract commocn lines
        cut -f 1 -d " " data/${x}.en.tmp/text > data/${x}.de.tmp/reclist1
        cut -f 1 -d " " data/${x}.de.tmp/text > data/${x}.de.tmp/reclist2
        comm -12 data/${x}.de.tmp/reclist1 data/${x}.de.tmp/reclist2 > data/${x}.de.tmp/reclist

        for lang in en de; do
            reduce_data_dir.sh data/${x}.${lang}.tmp data/${x}.de.tmp/reclist data/${x}.${lang}
            utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${x}.${lang}
        done
        rm -rf data/${x}.*.tmp
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt18/st1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt18/st1/dump/${train_dev}/delta${do_delta}/storage \
          ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${case}.txt
nlsyms=data/lang_1spm/non_lang_syms_${case}.txt
bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${case}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1spm/

    echo "make a non-linguistic symbol list for all languages"
    grep sp1.0 data/${train_set_prefix}.*/text.${case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=$(wc -l < ${dict})
    grep sp1.0 data/${train_set_prefix}.*/text.${case} | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_1spm/input.txt
    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text.${case} --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json
    local/data2json.sh --feat ${feat_dt_dir}/feats.scp --text data/${train_dev}/text.${case} --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        if [ ${rtask} = "dev.de" ] || [ ${rtask} = "test.de" ]; then
            local/data2json.sh --feat ${feat_recog_dir}/feats.scp --text data/${rtask}/text.${case} --bpecode ${bpemodel}.model \
                data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
        else
            local/data2json.sh --feat ${feat_recog_dir}/feats.scp --no_text true \
                data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json
        fi
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/$(echo ${x} | cut -f 1 -d ".").en
        local/update_json.sh --text ${data_dir}/text.${case} --bpecode ${bpemodel}.model \
            ${feat_dir}/data_${bpemode}${nbpe}.${case}.json ${data_dir} ${dict}
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${asr_model}" ]; then
        expname=${expname}_asrtrans
    fi
    if [ -n "${mt_model}" ]; then
        expname=${expname}_mttrans
    fi
else
    expname=${train_set}_${case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

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
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${case}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${case}.json \
        --asr-model ${asr_model} \
        --mt-model ${mt_model}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ST models
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
    nj=16

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.${case}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        if [ ${rtask} = "dev.de" ] || [ ${rtask} = "test.de" ]; then
            score_bleu.sh --case ${case} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
                ${expdir}/${decode_dir} de ${dict}
        else
            set=$(echo ${rtask} | cut -f 1 -d ".")
            local/score_bleu_reseg.sh --case ${case} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
                ${expdir}/${decode_dir} ${dict} ${st_ted} ${set}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
