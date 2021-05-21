#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=0         # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
nj=4            # number of parallel jobs for decoding
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config=
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
tgt_case=lc.rm
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
fsc=/project/ocean/sdalmia/fall_2020/projects/fsc_dataset/fluent_speech_commands_dataset

# bpemode (unigram or bpe)
nbpe=200
#nbpe=50
bpemode=bpe
# NOTE: nbpe=53 means character-level ST (lc.rm)
# NOTE: nbpe=66 means character-level ST (lc)
# NOTE: nbpe=98 means character-level ST (tc)

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=valid
train_test=test
recog_set="valid"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for x in $train_dev $train_set $train_test; do
        local/fsc_data_prep.sh ${fsc} ${x} /tmp/${x}
        mv /tmp/${x} data/${x}
        local/normalize_transcript.sh data/${x}
    done
fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3
    utils/combine_data.sh --extra-files "utt2uniq text.slu text.lc.rm" data/train_sp data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3
    utils/fix_data_dir.sh --utt_extra_files "text.slu text.lc.rm" data/train_sp
fi

train_set=train_sp
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    fbankdir=fbank
    for x in $train_dev $train_set $train_test; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh --utt_extra_files "text.slu text.lc.rm" data/${x}
        utils/validate_data_dir.sh data/${x}
    done

    for x in $train_dev $train_set $train_test; do
        remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x} data/${x}.trim
        rm -rf data/${x}
        mv data/${x}.trim data/${x}
    done

    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 16 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    for ttask in ${train_test}; do
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}; mkdir -p ${feat_trans_dir}
        dump.sh --cmd "$train_cmd" --nj 16 --do_delta $do_delta \
            data/${ttask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/trans/${ttask} \
            ${feat_trans_dir}
    done

fi

dict_slu=data/lang_1spm/${train_set}_slu_units.txt
echo "dictionary: ${dict_slu}"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1spm/

    echo "make a joint source and target dictionary"
    echo "<unk> 1" > ${dict_slu} # <unk> must be 1, 0 will be used for "blank" in CTC

    offset=$(wc -l < ${dict_slu})

    grep sp1.0 data/${train_set}/text.slu | cut -f 2- -d' ' | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict_slu}

    wc -l ${dict_slu}
fi

dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${tgt_case}.txt
nlsyms=data/lang_1spm/${train_set}_non_lang_syms_${tgt_case}.txt
bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${tgt_case}
echo "dictionary: ${dict}"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1spm/

    echo "make a non-linguistic symbol list for all languages"
    touch ${nlsyms}

    echo "make a joint source and target dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC

    offset=$(wc -l < ${dict})
    grep sp1.0 data/${train_set}/text.${tgt_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_1spm/input.txt
    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}

    wc -l ${dict}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    for x in $train_dev $train_set $train_test; do
        feat_dir=${dumpdir}/${x}/delta${do_delta};
        data2json.sh --nj 16 --feat ${feat_dir}/feats.scp --trans-type wrd --text data/${x}/text.slu data/${x} ${dict_slu} > ${feat_dir}/data_slu_asr${bpemode}${nbpe}.${src_case}_${tgt_case}.json
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

    for x in $train_dev $train_set $train_test; do
        feat_dir=${dumpdir}/${x}/delta${do_delta};
        update_json.sh --text data/${x}/text.${src_case} --bpecode ${bpemodel}.model \
            ${feat_dir}/data_slu_asr${bpemode}${nbpe}.${src_case}_${tgt_case}.json data/${x} ${dict}
    done
fi
