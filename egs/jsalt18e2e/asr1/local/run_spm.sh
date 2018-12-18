#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a baseline for "JSALT'18 Multilingual End-to-end ASR for Incomplete Data"
# We use 5 Babel language (Assamese Tagalog Swahili Lao Zulu), Librispeech (English), and CSJ (Japanese)
# as a target language, and use 10 Babel language (Cantonese Bengali Pashto Turkish Vietnamese
# Haitian Tamil Kurmanji Tok-Pisin Georgian) as a non-target language.
# The recipe first build language-independent ASR by using non-target languages

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network archtecture
# encoder related
etype=vggblstm     # encoder architecture type
elayers=4
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
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
batchsize=50
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=20

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# bpemode (unigram or bpe)
nbpe=8000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

# data set
# non-target languages: cantonese bengali pashto turkish vietnamese haitian tamil kurmanji tokpisin georgian
train_set=tr_babel10
train_dev=dt_babel10
# non-target
recog_set="dt_babel_cantonese et_babel_cantonese dt_babel_bengali et_babel_bengali dt_babel_pashto et_babel_pashto dt_babel_turkish et_babel_turkish\
 dt_babel_vietnamese et_babel_vietnamese dt_babel_haitian et_babel_haitian\
 dt_babel_tamil et_babel_tamil dt_babel_kurmanji et_babel_kurmanji dt_babel_tokpisin et_babel_tokpisin dt_babel_georgian et_babel_georgian"
# target
recog_set="dt_babel_assamese et_babel_assamese dt_babel_tagalog et_babel_tagalog dt_babel_swahili et_babel_swahili dt_babel_lao et_babel_lao dt_babel_zulu et_babel_zulu
 dt_csj_japanese et_csj_japanese_1 et_csj_japanese_2 et_csj_japanese_3\
 dt_libri_english_clean dt_libri_english_other et_libri_english_clean et_libri_english_other"
# whole set
recog_set="dt_babel_cantonese et_babel_cantonese dt_babel_assamese et_babel_assamese dt_babel_bengali et_babel_bengali dt_babel_pashto et_babel_pashto dt_babel_turkish et_babel_turkish\
 dt_babel_vietnamese et_babel_vietnamese dt_babel_haitian et_babel_haitian dt_babel_swahili et_babel_swahili dt_babel_lao et_babel_lao dt_babel_tagalog et_babel_tagalog\
 dt_babel_tamil et_babel_tamil dt_babel_kurmanji et_babel_kurmanji dt_babel_zulu et_babel_zulu dt_babel_tokpisin et_babel_tokpisin dt_babel_georgian et_babel_georgian\
 dt_csj_japanese et_csj_japanese_1 et_csj_japanese_2 et_csj_japanese_3\
 dt_libri_english_clean dt_libri_english_other et_libri_english_clean et_libri_english_other"

# subset options
# select the number of speakers for subset training experiments. (e.g. 1000; select 1000 speakers). Default: select the whole train set.
subset_num_spk=""

. utils/parse_options.sh || exit 1;

# data set
train_set=tr_babel10${subset_num_spk:+_${subset_num_spk}spk}

# data directories
csjdir=../../csj
libridir=../../librispeech
babeldir=../../babel

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ]; then
    # TODO
    # add a check whether the following data preparation is completed or not

    # CSJ Japanese
    if [ ! -d "$csjdir/asr1/data" ]; then
	echo "run $csjdir/asr1/run.sh first"
	exit 1
    fi
    lang_code=csj_japanese
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/train_nodup data/tr_${lang_code}
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/train_dev   data/dt_${lang_code}
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval1       data/et_${lang_code}_1
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval2       data/et_${lang_code}_2
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval3       data/et_${lang_code}_3
    # 1) change wide to narrow chars
    # 2) lower to upper chars
    for x in data/*${lang_code}*; do
        utils/copy_data_dir.sh ${x} ${x}_org
        cat ${x}_org/text | nkf -Z |\
            awk '{for(i=2;i<=NF;++i){$i = toupper($i)} print}' > ${x}/text
        rm -fr ${x}_org
    done

    # librispeech
    lang_code=libri_english
    if [ ! -d "$libridir/asr1/data" ]; then
	echo "run $libridir/asr1/run.sh first"
	exit 1
    fi
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/train_960  data/tr_${lang_code}
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/dev_clean  data/dt_${lang_code}_clean
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/dev_other  data/dt_${lang_code}_other
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/test_clean data/et_${lang_code}_clean
    utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/test_other data/et_${lang_code}_other

    # Babel
    for x in 101-cantonese 102-assamese 103-bengali 104-pashto 105-turkish 106-tagalog 107-vietnamese 201-haitian 202-swahili 203-lao 204-tamil 205-kurmanji 206-zulu 207-tokpisin 404-georgian; do
	langid=`echo $x | cut -f 1 -d"-"`
	lang_code=`echo $x | cut -f 2 -d"-"`
	if [ ! -d "$babeldir/asr1_${lang_code}/data" ]; then
	    echo "run $babeldir/asr1/local/run_all.sh first"
	    exit 1
	fi
        utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../babel/asr1_${lang_code}/data/train          data/tr_babel_${lang_code}
        utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../babel/asr1_${lang_code}/data/dev            data/dt_babel_${lang_code}
        utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../babel/asr1_${lang_code}/data/eval_${langid} data/et_babel_${lang_code}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}_${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}_${train_set}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then

    utils/combine_data.sh data/tr_babel10_org data/tr_babel_cantonese data/tr_babel_bengali data/tr_babel_pashto data/tr_babel_turkish data/tr_babel_vietnamese data/tr_babel_haitian data/tr_babel_tamil data/tr_babel_kurmanji data/tr_babel_tokpisin data/tr_babel_georgian
    utils/combine_data.sh data/dt_babel10_org data/dt_babel_cantonese data/dt_babel_bengali data/dt_babel_pashto data/dt_babel_turkish data/dt_babel_vietnamese data/dt_babel_haitian data/dt_babel_tamil data/dt_babel_kurmanji data/dt_babel_tokpisin data/dt_babel_georgian

    if [ ! -z $subset_num_spk ]; then
        # create a trainng subset with ${subset_num_spk} speakers (in total 7470)
        head -n $subset_num_spk <(utils/shuffle_list.pl data/tr_babel10_org/spk2utt | awk '{print $1}') > data/tr_babel10_org/spk_list_${subset_num_spk}spk
        utils/subset_data_dir.sh \
        --spk-list data/tr_babel10_org/spk_list_${subset_num_spk}spk \
        data/tr_babel10_org data/${train_set}_org
    fi

    # remove utt having more than 3000 frames or less than 10 frames or
    # remove utt having more than 400 characters or no more than 0 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/jsalt18e2e/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/jsalt18e2e/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    [ ! -d ${feat_tr_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set}_${train_set} ${feat_tr_dir}
    [ ! -d ${feat_dt_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev}_${train_set} ${feat_dt_dir}
   for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        [ ! -d ${feat_recog_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask}_${train_set} \
            ${feat_recog_dir}
    done
fi
dict=data/lang_char/train_${bpemode}${nbpe}_units.txt
nlsyms=data/lang_char/non_lang_syms.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/

    echo "make a non-linguistic symbol list for all languages"
    cut -f 2- data/tr_*/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} --bpecode ${bpemodel}.model \
         data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} --bpecode ${bpemodel}.model \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model\
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp${expname}
mkdir -p ${expdir}

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
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
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json \
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
        --epochs ${epochs}
fi

if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            &
        wait

        score_sclite.sh --nlsyms ${nlsyms} --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi
