#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=-1       # start from -1 if you need to start from data download
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

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
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
lm_layers=2
lm_units=650
lm_opt=sgd        # or adam
lm_batchsize=1024 # batch size in LM training
lm_epochs=20      # if the data size is large, we can reduce this
lm_maxlen=150     # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

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

train_set=train_trim
train_dev=dev_trim
recog_set="dev test"

if [ "${stage}" -le -1 ]; then
    echo "stage -1: Data Download"
    local/download_data.sh
fi

if [ "${stage}" -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/prepare_data.sh
    for dset in dev test train; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 "data/${dset}.orig" "data/${dset}"
    done
fi

feat_tr_dir="${dumpdir}/${train_set}/delta${do_delta}"; mkdir -p "${feat_tr_dir}"
feat_dt_dir="${dumpdir}/${train_dev}/delta${do_delta}"; mkdir -p "${feat_dt_dir}"
if [ "${stage}" -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in test dev train; do
        steps/make_fbank_pitch.sh --cmd "${train_cmd}" --nj 32 --write_utt2num_frames true \
            "data/${x}" "exp/make_fbank/${x}" "${fbankdir}"
    done

    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 400 characters or no more than 0 characters
    remove_longshortdata.sh --maxchars 400 data/train "data/${train_set}"
    remove_longshortdata.sh --maxchars 400 data/dev "data/${train_dev}"

    # compute global CMVN
    compute-cmvn-stats scp:"data/${train_set}/feats.scp" "data/${train_set}/cmvn.ark"

    # dump features for training
    dump.sh --cmd "${train_cmd}" --nj 32 --do_delta "${do_delta}" \
        "data/${train_set}/feats.scp" "data/${train_set}/cmvn.ark" exp/dump_feats/train "${feat_tr_dir}"
    dump.sh --cmd "${train_cmd}" --nj 32 --do_delta "${do_delta}" \
        "data/${train_dev}/feats.scp" "data/${train_set}/cmvn.ark" exp/dump_feats/dev "${feat_dt_dir}"
    for rtask in ${recog_set}; do
        feat_recog_dir="${dumpdir}/${rtask}/delta${do_delta}"; mkdir -p "${feat_recog_dir}"
        dump.sh --cmd "${train_cmd}" --nj 32 --do_delta "${do_delta}" \
            "data/${rtask}/feats.scp" "data/${train_set}/cmvn.ark" "exp/dump_feats/recog/${rtask}" \
            "${feat_recog_dir}"
    done
fi

dict="data/lang_1char/${train_set}_units.txt"
echo "dictionary: ${dict}"
if [ "${stage}" -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > "${dict}" # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 "data/${train_set}/text" | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> "${dict}"
    wc -l "${dict}"

    # make json labels
    data2json.sh --feat "${feat_tr_dir}/feats.scp" \
         "data/${train_set}" "${dict}" > "${feat_tr_dir}/data.json"
    data2json.sh --feat "${feat_dt_dir}/feats.scp" \
         "data/${train_dev}" "${dict}" > "${feat_dt_dir}/data.json"
    for rtask in ${recog_set}; do
        feat_recog_dir="${dumpdir}/${rtask}/delta${do_delta}"
        data2json.sh --feat "${feat_recog_dir}/feats.scp" \
            "data/${rtask}" "${dict}" > "${feat_recog_dir}/data.json"
    done
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_adim${adim}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}"
    if "${do_delta}"; then
        expdir="${expdir}_delta"
    fi
else
    expdir="exp/${train_set}_${backend}_${tag}"
fi
mkdir -p "${expdir}"

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z "${lmtag}" ]; then
    lmtag="${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}"
fi
lmexpdir="exp/train_rnnlm_${backend}_${lmtag}"
mkdir -p "${lmexpdir}"

if [ "${stage}" -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train
    [ ! -e ${lmdatadir} ] && mkdir -p "${lmdatadir}"
    gunzip -c db/TEDLIUM_release2/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py \
        | text2token.py -n 1 \
        > "${lmdatadir}/train.txt"
    text2token.py -s 1 -n 1 data/dev/text | cut -f 2- -d" " \
        > "${lmdatadir}/valid.txt"
    # use only 1 gpu
    if [ "${ngpu}" -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    "${cuda_cmd}"  --gpu "${ngpu}" "${lmexpdir}/train.log" \
        lm_train.py \
        --ngpu "${ngpu}" \
        --backend "${backend}" \
        --verbose 1 \
        --outdir "${lmexpdir}" \
        --train-label "${lmdatadir}/train.txt" \
        --valid-label "${lmdatadir}/valid.txt" \
        --resume "${lm_resume}" \
        --layer "${lm_layers}" \
        --unit "${lm_units}" \
        --opt "${lm_opt}" \
        --batchsize "${lm_batchsize}" \
        --epoch "${lm_epochs}" \
        --maxlen "${lm_maxlen}" \
        --dict "${dict}"
fi

if [ "${stage}" -le 4 ]; then
    echo "stage 4: Network Training"
    "${cuda_cmd}"  --gpu "${ngpu}" "${expdir}/train.log" \
        asr_train.py \
        --ngpu "${ngpu}" \
        --backend "${backend}" \
        --outdir "${expdir}/results" \
        --debugmode "${debugmode}" \
        --dict "${dict}" \
        --debugdir "${expdir}" \
        --minibatches "${N}" \
        --verbose "${verbose}" \
        --resume "${resume}" \
        --train-json "${feat_tr_dir}/data.json" \
        --valid-json "${feat_dt_dir}/data.json" \
        --etype "${etype}" \
        --elayers "${elayers}" \
        --eunits "${eunits}" \
        --eprojs "${eprojs}" \
        --subsample "${subsample}" \
        --dlayers "${dlayers}" \
        --dunits "${dunits}" \
        --atype "${atype}" \
        --adim "${adim}" \
        --aconv-chans "${aconv_chans}" \
        --aconv-filts "${aconv_filts}" \
        --mtlalpha "${mtlalpha}" \
        --batch-size "${batchsize}" \
        --maxlen-in "${maxlen_in}" \
        --maxlen-out "${maxlen_out}" \
        --sampling-probability "${samp_prob}" \
        --opt "${opt}" \
        --epochs "${epochs}"
fi

if [ "${stage}" -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir="decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}"
        feat_recog_dir="${dumpdir}/${rtask}/delta${do_delta}"

        # split data
        splitjson.py --parts "${nj}" "${feat_recog_dir}/data.json"

        #### use CPU for decoding
        ngpu=0

        "${decode_cmd}" JOB=1:"${nj}" "${expdir}/${decode_dir}/log/"decode.JOB.log \
            asr_recog.py \
            --ngpu "${ngpu}" \
            --backend "${backend}" \
            --debugmode "${debugmode}" \
            --verbose "${verbose}" \
            --recog-json "${feat_recog_dir}/split${nj}utt/"data.JOB.json \
            --result-label "${expdir}/${decode_dir}/"data.JOB.json \
            --model "${expdir}/results/${recog_model}" \
            --beam-size "${beam_size}" \
            --penalty "${penalty}" \
            --maxlenratio "${maxlenratio}" \
            --minlenratio "${minlenratio}" \
            --ctc-weight "${ctc_weight}" \
            --rnnlm "${lmexpdir}/rnnlm.model.best" \
            --lm-weight "${lm_weight}" &
        wait

        score_sclite.sh --wer true "${expdir}/${decode_dir}" "${dict}"

    ) &
    done
    wait
    echo "Finished"
fi

