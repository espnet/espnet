#!/usr/bin/env bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=0         # start from -1 if you need to start from data download
stop_stage=100
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

# bpemode (unigram or bpe)
nbpe=8000
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
how2_dir=../../how2
mustc_dir=../../must_c

train_set=train
train_dev=dev
trans_set="et_must_c_tst-COMMON.pt et_must_c_tst-HE.pt et_how2_dev5.pt"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"

    # Must-C
    if [ ! -d "${mustc_dir}/st1/data/train_sp.en-pt.pt" ]; then
        echo "run ${mustc_dir}/st1/run.sh first"
        exit 1
    fi
    data_code=mustc
    for lang in pt en; do
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/train_sp.en-pt.${lang}   data/tr_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/dev.en-pt.${lang}        data/dt_${data_code}.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/tst-COMMON.en-pt.${lang} data/et_${data_code}_tst-COMMON.${lang}
        local/copy_data_dir.sh --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${mustc_dir}/st1/data/tst-HE.en-pt.${lang}     data/et_${data_code}_tst-HE.${lang}
    done

    # How2
    if [ ! -d "${how2_dir}/st1/data/train.pt" ]; then
        echo "run ${how2_dir}/st1/run.sh first"
        exit 1
    fi
    data_code=how2
    for lang in pt en; do
        local/copy_data_dir.sh --validate_opts --no-wav --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${how2_dir}/st1/data/train.${lang} data/tr_${data_code}.${lang}
        local/copy_data_dir.sh --validate_opts --no-wav --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${how2_dir}/st1/data/val.${lang}   data/dt_${data_code}.${lang}
        local/copy_data_dir.sh --validate_opts --no-wav --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${how2_dir}/st1/data/dev5.${lang}  data/et_${data_code}_dev5.${lang}
    done
    local/copy_data_dir.sh --validate_opts --no-wav --utt-prefix ${data_code}- --spk-prefix ${data_code}- ${how2_dir}/asr1/data/test_set_iwslt2019.en data/et_${data_code}_test_set_iwslt2019.en
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by 40-dimensional fbanks with pitch on each frame
    # to unify the fbank feature setup with how2
    # This is different from the other ESPnet recipe (80-dimensional fbanks)
    for x in tr_must_c.pt dt_must_c.pt $(echo ${trans_set} | tr ' ' '\n' | grep -v how2); do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    rm data/*/segments
    utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/${train_set} data/tr_must_c.pt data/tr_how2.pt
    utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/${train_set}.en data/tr_must_c.en data/tr_how2.en
    utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/${train_dev} data/dt_must_c.pt data/dt_how2.pt
    utils/combine_data.sh --extra_files "text.tc text.lc text.lc.rm" data/${train_dev}.en data/dt_must_c.en data/dt_how2.en

    echo "Remove offlimit"
    cp -rf data/${train_set} data/${train_set}.tmp
    cp data/${train_set}/utt2spk data/${train_set}/utt2spk.org
    local/filter_offlimit.py --offlimit_list local/offlimit_list --utt2spk data/${train_set}/utt2spk.org > data/${train_set}/utt2spk
    utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt19/st1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt19/st1/dump/${train_dev}/delta${do_delta}/storage \
          ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    for x in ${trans_set}; do
        feat_trans_dir=${dumpdir}/${x}/delta${do_delta}; mkdir -p ${feat_trans_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${x}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/trans/${x} \
            ${feat_trans_dir}
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
    { grep sp1.0 data/${train_set}/text.tc | cut -f 2- -d' ';
      grep sp1.0 data/${train_set}.en/text.tc | cut -f 2- -d' ';
      grep how2 data/${train_set}/text.tc | cut -f 2- -d' ';
      grep how2 data/${train_set}.en/text.tc | cut -f 2- -d' ';
    } > data/lang_1spm/input.txt
    # NOTE: speed perturbation is not applied in how2
    grep -o -P '&[^;]*;' data/lang_1spm/input.txt | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a joint source and target dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=$(wc -l < ${dict})
    grep -v -e '^\s*$' data/lang_1spm/input.txt > data/lang_1spm/input.txt.tmp
    mv data/lang_1spm/input.txt.tmp data/lang_1spm/input.txt
    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text.tc --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --text data/${train_dev}/text.tc --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json
    for x in ${trans_set}; do
        feat_trans_dir=${dumpdir}/${x}/delta${do_delta}
        data2json.sh --feat ${feat_trans_dir}/feats.scp --text data/${x}/text.tc --bpecode ${bpemodel}.model \
            data/${x} ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}.json
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/${x}.en
        update_json.sh --text ${data_dir}/text.lc.rm --bpecode ${bpemodel}.model \
            ${feat_dir}/data_${bpemode}${nbpe}.json ${data_dir} ${dict}
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
    if [ -n "${asr_model}" ]; then
        expname=${expname}_asrtrans
    fi
    if [ -n "${mt_model}" ]; then
        expname=${expname}_mttrans
    fi
else
    expname=${train_set}_${backend}_${tag}
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
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json \
        --enc-init ${asr_model} \
        --dec-init ${mt_model}
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
    for x in ${trans_set}; do
    (
        decode_dir=decode_${x}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${x}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data_${bpemode}${nbpe}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            st_trans.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        score_bleu.sh --case tc --bpemodel ${bpemodel}.model \
            ${expdir}/${decode_dir} pt ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
