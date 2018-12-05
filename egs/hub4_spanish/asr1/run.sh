#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

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
mtlalpha=0.5

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# decoding parameter
beam_size=20
nbest=1
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.1
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
recog_set="eval"

# data
audio_data=/export/corpora/LDC/LDC98S74
transcript_data=/export/corpora/LDC/LDC98T29
eval_data=/export/corpora/LDC/LDC2001S91
dev_list=dev.list


if [ "${stage}" -le 0 ]; then
  # Eval dataset preparation
  # prepare_data.sh does not really care about the order or number of the
  # corpus directories
  local/prepare_data.sh \
    "${eval_data}/HUB4_1997NE/doc/h4ne97sp.sgm" \
    "${eval_data}/HUB4_1997NE/h4ne_sp/h4ne97sp.sph" data/eval
  local/prepare_test_text.pl \
    "<unk>" data/eval/text > data/eval/text.clean
  mv data/eval/text data/eval/text.old
  mv data/eval/text.clean data/eval/text
  utils/fix_data_dir.sh data/eval

  ## Training dataset preparation
  local/prepare_data.sh "${audio_data}" "${transcript_data}" data/train
  local/prepare_training_text.pl \
    "<unk>" data/train/text > data/train/text.clean
  mv data/train/text data/train/text.old
  mv data/train/text.clean data/train/text
  utils/fix_data_dir.sh data/train

  # For generating the dev set. Use provided utterance list otherwise
  # num_dev=$(cat data/eval/segments | wc -l)
  # ./utils/subset_data_dir.sh data/train ${num_dev} data/dev

  ./utils/subset_data_dir.sh --utt-list "${dev_list}" data/train data/dev

  mv data/train data/train.tmp
  mkdir -p data/train
  awk '{print $1}' data/dev/segments | grep -vf - data/train.tmp/segments > data/train/uttlist.list
  ./utils/subset_data_dir.sh --utt-list data/train/uttlist.list data/train.tmp data/train
fi

feat_tr_dir="${dumpdir}/${train_set}/delta${do_delta}"; mkdir -p "${feat_tr_dir}"
feat_dt_dir="${dumpdir}/${train_dev}/delta${do_delta}"; mkdir -p "${feat_dt_dir}"
if [ "${stage}" -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${train_dev} ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "${train_cmd}" --nj 50 --write_utt2num_frames true \
            "data/${x}" "exp/make_fbank/${x}" "${fbankdir}"
    done

    # compute global CMVN
    compute-cmvn-stats scp:"data/${train_set}/feats.scp" "data/${train_set}/cmvn.ark"

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d "${feat_tr_dir}/storage" ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/"${USER}/espnet-data/egs/hub4_spanish/asr1/dump/${train_set}/delta${do_delta}/storage" \
        "${feat_tr_dir}/storage"
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d "${feat_dt_dir}/storage" ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/"${USER}/espnet-data/egs/hub4_spanish/asr1/dump/${train_dev}/delta${do_delta}/storage" \
        "${feat_dt_dir}/storage"
    fi
    dump.sh --cmd "${train_cmd}" --nj 32 --do_delta "${do_delta}" \
        "data/${train_set}/feats.scp" "data/${train_set}/cmvn.ark" exp/dump_feats/train "${feat_tr_dir}"
    dump.sh --cmd "${train_cmd}" --nj 4 --do_delta "${do_delta}" \
        "data/${train_dev}/feats.scp" "data/${train_set}/cmvn.ark" exp/dump_feats/dev "${feat_dt_dir}"
    for rtask in ${recog_set}; do
        feat_recog_dir="${dumpdir}/${rtask}/delta${do_delta}"; mkdir -p "${feat_recog_dir}"
        dump.sh --cmd "${train_cmd}" --nj 4 --do_delta "${do_delta}" \
            "data/${rtask}/feats.scp" "data/${train_set}/cmvn.ark" "exp/dump_feats/recog/${rtask}" \
            "${feat_recog_dir}"
    done
fi

dict="data/lang_1char/${train_set}_units.txt"
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ "${stage}" -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- "data/${train_set}/text" | tr " " "\n" | sort | uniq | grep "<" > "${nlsyms}"
    cat "${nlsyms}"

    echo "make a dictionary"
    echo "<unk> 1" > "${dict}" # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l "${nlsyms}" "data/${train_set}/text" | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> "${dict}"
    wc -l "${dict}"

    echo "make json files"
    data2json.sh --feat "${feat_tr_dir}/feats.scp" --nlsyms "${nlsyms}" \
         "data/${train_set}" "${dict}" > "${feat_tr_dir}/data.json"
    data2json.sh --feat "${feat_dt_dir}/feats.scp" --nlsyms "${nlsyms}" \
         "data/${train_dev}" "${dict}" > "${feat_dt_dir}/data.json"
    for rtask in ${recog_set}; do
        feat_recog_dir="${dumpdir}/${rtask}/delta${do_delta}"
        data2json.sh --feat "${feat_recog_dir}/feats.scp" \
            --nlsyms "${nlsyms}" "data/${rtask}" "${dict}" > "${feat_recog_dir}/data.json"
    done
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}"
    if [ "${lsm_type}" != "" ]; then
        expdir="${expdir}_lsm${lsm_type}${lsm_weight}"
    fi
    if "${do_delta}"; then
        expdir="${expdir}_delta"
    fi
else
    expdir="exp/${train_set}_${backend}_${tag}"
fi
mkdir -p "${expdir}"

if [ "${stage}" -le 3 ]; then
    echo "stage 3: Network Training"

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
        --seed "${seed}" \
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
        --awin "${awin}" \
        --aheads "${aheads}" \
        --aconv-chans "${aconv_chans}" \
        --aconv-filts "${aconv_filts}" \
        --mtlalpha "${mtlalpha}" \
        --lsm-type "${lsm_type}" \
        --lsm-weight "${lsm_weight}" \
        --batch-size "${batchsize}" \
        --maxlen-in "${maxlen_in}" \
        --maxlen-out "${maxlen_out}" \
        --sampling-probability "${samp_prob}" \
        --opt "${opt}" \
        --epochs "${epochs}"
fi

if [ "${stage}" -le 4 ]; then
    echo "stage 4: Decoding"
    nj=32

    for rtask in ${recog_set} ${train_dev}; do
    (
        decode_dir="decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}"
        feat_recog_dir="${dumpdir}/${rtask}/delta${do_delta}"

        # split data
        splitjson.py --parts "${nj}" "${feat_recog_dir}/data.json"

        #### use CPU for decoding
        ngpu=0

        "${decode_cmd}" JOB=1:"${nj}" "${expdir}/${decode_dir}/log/"decode.JOB.log \
            asr_recog.py \
            --ngpu "${ngpu}" \
            --backend "${backend}" \
            --recog-json "${feat_recog_dir}/split${nj}utt/"data.JOB.json \
            --result-label "${expdir}/${decode_dir}/"data.JOB.json \
            --model "${expdir}/results/${recog_model}" \
            --beam-size "${beam_size}" \
            --nbest ${nbest} \
            --penalty "${penalty}" \
            --maxlenratio "${maxlenratio}" \
            --minlenratio "${minlenratio}" \
            --ctc-weight "${ctc_weight}" &
        wait

        score_sclite.sh --wer true --nlsyms "${nlsyms}" "${expdir}/${decode_dir}" "${dict}"

    ) &
    done
    wait
    echo "Finished"
fi
