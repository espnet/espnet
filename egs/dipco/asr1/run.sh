#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
dumpdir=dump   # directory to dump full features

# feature configuration
do_delta=false

train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10

# exp tag
tag="" # tag for managing experiments.

dipco_corpus=${PWD}/db/DiPCo
chime5_dir=${PWD}/../../chime5/asr1

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_data.sh
fi

json_dir=${dipco_corpus}/transcriptions
audio_dir=${dipco_corpus}/audio

enhancement=beamformit

train_set=train_worn_u200k  # based on CHiME5 recipe
train_dev=dev_${enhancement}_ref
recog_set="dev_worn dev_${enhancement}_ref eval_worn eval_${enhancement}_ref"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    mictype=worn
    for dset in dev eval; do
        local/prepare_data.sh --mictype ${mictype} \
            ${audio_dir}/${dset} ${json_dir}/${dset} \
            data/${dset}_${mictype}
    done

    enhandir=enhan
    for dset in dev eval; do
        for mictype in u01 u02 u03 u04 u05; do
            local/run_beamformit.sh --cmd "$train_cmd" --bmf "1 2 3 4 5 6 7"\
                        ${audio_dir}/${dset} \
                        ${enhandir}/${dset}_${enhancement}_${mictype} \
                        ${mictype} &
        done
        wait
    done

    for dset in dev eval; do
        # The ref mic is the same as the worn: close-talk
        for mictype in u01 u02 u03 u04 u05; do
            local/prepare_data.sh --mictype ${mictype} "$PWD/${enhandir}/${dset}_${enhancement}_${mictype}" \
                    ${json_dir}/${dset} data/${dset}_${enhancement}_ref_${mictype}
        done
        ddirs=$(ls -d data/${dset}_${enhancement}_ref_u0*)
        utils/combine_data.sh data/${dset}_${enhancement}_ref ${ddirs}
        rm -rf data/${dset}_${enhancement}_ref_u0*
    done
    # only use left channel for worn mic recognition
    # you can use both left and right channels for training
    for dset in dev eval; do
        utils/copy_data_dir.sh data/${dset}_worn data/${dset}_worn_stereo
        grep "\.L-" data/${dset}_worn_stereo/text > data/${dset}_worn/text
        utils/fix_data_dir.sh data/${dset}_worn
    done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done
    # compute global CMVN
    # Use CHiME 5 CMVN
    # compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
fi

dict=${chime5_dir}/data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_dev}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
    cat ${nlsyms}

    # echo "make a dictionary"
    # echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    # text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    # | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    # wc -l ${dict}

    # # remove 1 or 0 length outputs
    # utils/copy_data_dir.sh data/train_worn_u200k data/train_worn_u200k_org
    # remove_longshortdata.sh --nlsyms ${nlsyms} --minchars 1 data/train_worn_u200k_org data/train_worn_u200k

    # dump features
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data/${rtask}/feats.scp ${chime5_dir}/data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_recog_dir}
    done

    echo "make json files"
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
if [ -d ${chime5_dir}/${lmexpdir} ]; then
    if [ ! -e ${lmexpdir} ]; then
        ln -s ${chime5_dir}/${lmexpdir} ${lmexpdir}
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    echo "The LM model should be obtained from CHiME5 recipe"
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}

if [ -d ${chime5_dir}/${expdir} ]; then
    if [ ! -e ${expdir} ]; then
        ln -s ${chime5_dir}/${expdir} ${expdir}
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    echo "The LM model should be obtained from CHiME5 recipe"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_dipco_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
