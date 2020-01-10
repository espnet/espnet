#!/bin/bash

# Copyright 2020 ARL (John Morgan)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0
stop_stage=100

ngpu=2 # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=             # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10

# data
data=/mnt/corpora/LDC2006S37
data_url=www.openslr.org/resources/39/LDC2006S37.tar.gz

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp
train_dev=devtest
recog_set="devtest test native nonnative"
fbankdir=fbank
lmdatadir=data/local/lm_train
dict=data/lang_1char/${train_set}_units.txt


if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    echo "$0 Stage -1: Data download."
    wget $data_url
    gunzip LDC2006S37.tar.gz
    tar -xvf LDC2006S37.tar
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "$0 Stage 0: Data preparation."
    local/prepare_data.sh $data
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "$0: stage 1: Feature Generation."
    for f in devtest native nonnative test train; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 \
            --write_utt2num_frames true data/$f exp/make_fbank/$f ${fbankdir} || exit 1;
        utils/fix_data_dir.sh data/$f || exit 1;
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "$0: Stage 2: Speed-perturb."
    utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/$train_set data/temp1 data/temp2 data/temp3
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "$0: Stage 3: Filterbanks with pitch."
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 \
        --write_utt2num_frames true data/$train_set exp/make_fbank/$train_set \
        $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$train_set || exit 1;

    echo "$0: compute global CMVN."
    compute-cmvn-stats scp:data/$train_set/feats.scp data/$train_set/cmvn.ark
fi

feat_tr_dir=$dumpdir/${train_set}/delta${do_delta}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "$0 Stage 4: Dump features for training."
    mkdir -vp $feat_tr_dir
    mkdir -vp $feat_dt_dir
    split_dir=$(echo $PWD | awk -F "/" '{print $NF "/" $(NF-1)}')
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/$train_set/feats.scp data/$train_set/cmvn.ark exp/dump_feats/train \
        $feat_tr_dir || exit 1;
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    for rtask in $recog_set; do
        echo "$0 Stage 5: Dumping for task $rtask."
        feat_recog_dir=$dumpdir/$rtask/delta${do_delta}
    mkdir -vp $feat_recog_dir
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
        data/$rtask/feats.scp data/$train_set/cmvn.ark \
        exp/dump_feats/recog/$rtask $feat_recog_dir || exit 1;
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    #check non-linguistic symbols used in the corpus.
    echo "$0 Stage 6: Dictionary   Preparation."
    mkdir -vp data/lang_1char
    echo "<unk> 1" > $dict # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/$train_set/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> $dict
    wc -l $dict
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "$0 Stage 7: Make json files."
    data2json.sh --feat $feat_tr_dir/feats.scp --filetype mat \
        data/$train_set $dict > $feat_tr_dir/data.json
    for rtask in $recog_set; do
        feat_recog_dir=$dumpdir/$rtask/delta${do_delta}
        data2json.sh --feat $feat_recog_dir/feats.scp \
            data/$rtask $dict > $feat_recog_dir/data.json
    done
fi

# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z $lmtag ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/$lmexpname
mkdir -vp $lmexpdir

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "$0 Stage 8: LM Preparation"
    mkdir -vp $lmdatadir
    text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
        > $lmdatadir/train.txt
    text2token.py -s 1 -n 1 data/$train_dev/text | cut -f 2- -d" " \
        > $lmdatadir/valid.txt

    ${cuda_cmd} --gpu $ngpu $lmexpdir/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend $backend \
        --verbose 1 \
        --outdir $lmexpdir \
        --tensorboard-dir tensorboard/$lmexpname \
        --train-label $lmdatadir/train.txt \
        --valid-label $lmdatadir/valid.txt \
        --resume $lm_resume \
        --dict $dict
fi

if [ -z $tag ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/$expname
mkdir -vp $expdir

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    echo "$0 Stage 9: Network Training."
    ${cuda_cmd} --gpu $ngpu $expdir/train.log \
    asr_train.py \
    --config $train_config \
    --ngpu $ngpu \
    --backend $backend \
    --outdir $expdir/results \
    --tensorboard-dir tensorboard/$expname \
    --debugmode $debugmode \
    --dict $dict \
    --debugdir $expdir \
    --minibatches $N \
    --verbose $verbose \
    --resume $resume \
    --train-json $feat_tr_dir/data.json \
    --valid-json $feat_dt_dir/data.json
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    echo "$0 Stage 10: Decoding."
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend $backend \
    --snapshots $expdir/results/snapshot.ep.* \
    --out $expdir/results/$recog_model \
    --num $n_average
    fi
    pids=() # initialize pids
    for rtask in $recog_set; do
        (
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
            feat_recog_dir=$dumpdir/$rtask/delta${do_delta}

        echo "$0: Split data."
        splitjson.py --parts $nj $feat_recog_dir/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:$nj $expdir/$decode_dir/log/decode.JOB.log \
            asr_recog.py \
            --config $decode_config \
            --ngpu $ngpu \
            --backend $backend \
            --batchsize 0 \
            --recog-json $feat_recog_dir/split${nj}utt/data.JOB.json \
            --result-label $expdir/$decode_dir/data.JOB.json \
            --model $expdir/results/$recog_model  \
            --rnnlm $lmexpdir/rnnlm.model.best

        score_sclite.sh $expdir/$decode_dir $dict

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        echo "Finished"
fi
