#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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
drop=0.3
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
opt=adadelta
epochs=20
patience=3

# decoding parameter
beam_size=20
penalty=0.2
maxlenratio=0.8
minlenratio=0
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
sfisher_speech=/export/corpora/LDC/LDC2010S01
sfisher_transcripts=/export/corpora/LDC/LDC2010T04
split=local/splits/split_fisher

callhome_speech=/export/corpora/LDC/LDC96S35
callhome_transcripts=/export/corpora/LDC/LDC96T17
split_callhome=local/splits/split_callhome

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

train_set=train.en
train_dev=dev.en
recog_set="fisher_dev.en fisher_dev2.en fisher_test.en callhome_devtest.en callhome_evltest.en"

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/fsp_data_prep.sh $sfisher_speech $sfisher_transcripts
    local/callhome_data_prep.sh $callhome_speech $callhome_transcripts

    # split data
    local/create_splits.sh $split
    local/callhome_create_splits.sh $split_callhome

    # concatenate multiple utterances
    local/normalize_trans.sh $sfisher_transcripts $callhome_transcripts
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in fisher_train fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest; do
        # upsample audio from 8k to 16k to make a recipe consistent with others
        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp

        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/fisher_train data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/fisher_train data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/fisher_train data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/train_sp data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        data/train_sp exp/make_fbank/train_sp ${fbankdir}
    for lang in es en; do
        cat data/fisher_train/utt2spk | awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' > data/train_sp/utt_map
        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/fisher_train/text.${lang} >data/train_sp/text.${lang}
        cat data/fisher_train/utt2spk | awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' > data/train_sp/utt_map
        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/fisher_train/text.${lang} >>data/train_sp/text.${lang}
        cat data/fisher_train/utt2spk | awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' > data/train_sp/utt_map
        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/fisher_train/text.${lang} >>data/train_sp/text.${lang}
    done

    # Divide into Es and En
    for x in train_sp fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest; do
        local/divide_lang.sh data/${x}
    done

    for lang in es en; do
        cp -rf data/train_sp.${lang} data/train.${lang}
        cp -rf data/fisher_dev.${lang} data/dev.${lang}
        # NOTE: do not use callhome_train for the training set
    done

    for x in train dev; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        for lang in es en; do
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
        done

        # Match the number of utterances between source and target languages
        # extract commocn lines
        cut -f -1 -d " " data/${x}.es.tmp/text > data/${x}.en.tmp/reclist1
        cut -f -1 -d " " data/${x}.en.tmp/text > data/${x}.en.tmp/reclist2
        comm -12 data/${x}.en.tmp/reclist1 data/${x}.en.tmp/reclist2 > data/${x}.en.tmp/reclist

        for lang in es en; do
            reduce_data_dir.sh data/${x}.${lang}.tmp data/${x}.en.tmp/reclist data/${x}.${lang}
            utils/fix_data_dir.sh data/${x}.${lang}
        done
        rm -rf data/${x}.*.tmp
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/fisher_callhome_spanish_st/st1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/fisher_callhome_spanish_st/st1/dump/${train_dev}/delta${do_delta}/storage \
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
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list for all languages"
    cut -f 2- -d " " data/train*/text | grep -o -P '&[^;]*;' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    # Share the same dictinary between source and target languages
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/train*/text | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d " " | tr " " "\n" \
      | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
        data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --nlsyms ${nlsyms} \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/`echo ${x} | cut -f -1 -d "."`.es
        local/update_json.sh --nlsyms ${nlsyms} ${feat_dir}/data.json ${data_dir} ${dict}
    done

    # Fisher has 4 references per utterance
    for rtask in fisher_dev.en fisher_dev2.en fisher_test.en; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        for no in 1 2 3; do
          local/data2json.sh --text data/${rtask}/text.${no} --feat ${feat_recog_dir}/feats.scp --nlsyms ${nlsyms} \
              data/${rtask} ${dict} > ${feat_recog_dir}/data_${no}.json
        done
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}${adim}_${opt}_sampprob${samp_prob}_lsm${lsm_weight}_drop${drop}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_wd${weight_decay}
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

if [ ${stage} -le 4 ]; then
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
        --dropout-rate ${drop} \
        --opt ${opt} \
        --epochs ${epochs} \
        --patience ${patience} \
        --weight-decay ${weight_decay} \
        --asr-model ${asr_model} \
        --mt-model ${mt_model}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

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
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            &
        wait

        # Fisher has 4 references per utterance
        if [ ${rtask} = "fisher_dev.en" ] || [ ${rtask} = "fisher_dev2.en" ] || [ ${rtask} = "fisher_test.en" ]; then
            for no in 1 2 3; do
              cp ${feat_recog_dir}/data_${no}.json ${expdir}/${decode_dir}/data_ref${no}.json
            done
        fi

        local/score_bleu.sh --set ${rtask} --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi
