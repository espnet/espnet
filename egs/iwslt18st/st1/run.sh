#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
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
etype=vggblstm     # encoder architecture type
elayers=5
eunits=1024
eprojs=1024
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=2
dunits=1024
# attention related
atype=add
adim=1024

# regualrization option
samp_prob=0.2
lsm_type=unigram
lsm_weight=0.1
drop_enc=0.3
drop_dec=0.0
weight_decay=0.000001

# transfer learning ralated
asr_model=
mt_model=
# NOTE: reserve here for the future usage

# minibatch related
batchsize=15
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
opt=adadelta
epochs=20
patience=3
eps_decay=0.01

# decoding parameter
beam_size=20
penalty=0.2
maxlenratio=0.8
minlenratio=0
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/b08/inaguma/IWSLT

# text normalization related
lc=true

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

train_set=train_nodev.de
train_dev=dev.de
recog_set="dev2010.de tst2010.de tst2013.de tst2014.de tst2015.de"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    for part in train dev2010 tst2010 tst2013 tst2014 tst2015; do
        local/download_and_untar.sh ${datadir} ${part}
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep_train.sh ${datadir}
    for part in dev2010 tst2010 tst2013 tst2014 tst2015; do
        local/data_prep_eval.sh ${datadir} ${part}
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
    for x in train dev2010 tst2010 tst2013 tst2014 tst2015; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    # Divide into source and target languages
    for x in train dev2010 tst2010 tst2013 tst2014 tst2015; do
        local/divide_lang.sh data/${x}
    done

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    for lang in en de; do
        remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
    done

    # Match the number of utterances between source and target languages
    # extract commocn lines
    cut -f -1 -d " " data/train.en.tmp/segments > data/train.de.tmp/reclist1
    cut -f -1 -d " " data/train.de.tmp/segments > data/train.de.tmp/reclist2
    comm -12 data/train.de.tmp/reclist1 data/train.de.tmp/reclist2 > data/train.de.tmp/reclist

    for lang in en de; do
        reduce_data_dir.sh data/train.${lang}.tmp data/train.de.tmp/reclist data/train.${lang}
        utils/fix_data_dir.sh data/train.${lang}
    done
    rm -rf data/${x}.*.tmp

    # make a dev set
    for lang in en de; do
        utils/subset_data_dir.sh --first data/train.${lang} 4000 data/dev.${lang}
        n=$(($(wc -l < data/train.${lang}/segments) - 4000))
        utils/subset_data_dir.sh --last data/train.${lang} ${n} data/train_nodev.${lang}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt18st/st1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/iwslt18st/st1/dump/${train_dev}/delta${do_delta}/storage \
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

dict=data/lang_1char/train_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list for all languages"
    cut -f 2- -d " " data/train_nodev.*/text | grep -o -P '&[^;]*;|@-@' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    # Share the same dictinary between source and target languages
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/train_nodev.*/text | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d " " | tr " " "\n" \
      | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
        data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        local/data2json.sh --feat ${feat_recog_dir}/feats.scp --nlsyms ${nlsyms} \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}${adim}_${opt}_sampprob${samp_prob}_lsm${lsm_weight}_drop${drop_enc}${drop_dec}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_wd${weight_decay}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ ! -z ${asr_model} ]; then
      expname=${expname}_asrtrans
    fi
    if [ ! -z ${mt_model} ]; then
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
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --mtlalpha 0 \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --dropout-rate ${drop_enc} \
        --dropout-rate-decoder ${drop_dec} \
        --opt ${opt} \
        --sortagrad ${sortagrad} \
        --eps-decay 1 \
        --epochs ${epochs} \
        --patience ${patience} \
        --eps-decay ${eps_decay} \
        --weight-decay ${weight_decay} \
        --asr-model ${asr_model} \
        --mt-model ${mt_model}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    pids=() # initialize pids
    # for rtask in ${train_dev} ${recog_set}; do
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}
        mkdir -p ${expdir}/${decode_dir}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio}

        if [ ${rtask} = "dev.de" ]; then
          local/score_bleu.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        else
          set=$(echo ${rtask} | cut -f -1 -d ".")
          local/score_bleu_reseg.sh --lc ${lc} --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict} ${set}
        fi

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
