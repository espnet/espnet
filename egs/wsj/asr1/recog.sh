#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
dumpdir=dump   # directory to dump full features

# feature configuration
do_delta=false

# network architecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.2

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_batchsize=300    # 1024 for character LMs
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=30
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# data
wav=""

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

train_set=train_si284
test_set=online

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ -z "${wav}" ]; then
        echo "Please specify --wav option"
	exit 1
    fi
    echo "stage 0: Data preparation"
    mkdir -p data/${test_set}
    base=`basename $wav .wav`
    echo "$base $wav" > data/${test_set}/wav.scp
    echo "xxxx $base" > data/${test_set}/spk2utt
    echo "$base xxxx" > data/${test_set}/utt2spk
    echo "$base <NOISE>" > data/${test_set}/text
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
        data/${test_set} exp/make_fbank/${test_set} ${fbankdir}

    feat_recog_dir=${dumpdir}/${test_set}/delta${do_delta}; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
        data/${test_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${test_set} \
        ${feat_recog_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"
    feat_recog_dir=${dumpdir}/${test_set}/delta${do_delta}
    data2json.sh --feat ${feat_recog_dir}/feats.scp \
        --nlsyms ${nlsyms} data/${test_set} ${dict} > ${feat_recog_dir}/data.json
fi

if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if [ "${lsm_type}" != "" ]; then
        expname=${expname}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"
    nj=1

    decode_dir=decode_${test_set}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
    if [ ${use_wordlm} = true ]; then
        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
    else
        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
    fi
    if [ ${lm_weight} == 0 ]; then
        recog_opts=""
    fi
    feat_recog_dir=${dumpdir}/${test_set}/delta${do_delta}

    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

    #### use CPU for decoding
    ngpu=0

    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/${recog_model} \
        --beam-size ${beam_size} \
        --penalty ${penalty} \
        --maxlenratio ${maxlenratio} \
        --minlenratio ${minlenratio} \
        --ctc-weight ${ctc_weight} \
        --lm-weight ${lm_weight} \
        ${recog_opts} &
    wait

    score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    echo ""
    echo ""
    recog_text=`cat ${expdir}/${decode_dir}/result.wrd.txt | grep HYP: | cut -c 7-`
    echo "Recognized text: ${recog_text}"
    echo ""
    echo ""

    echo "Finished"
fi
