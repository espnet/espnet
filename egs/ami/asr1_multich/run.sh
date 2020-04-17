#!/bin/bash

# Copyright 2019 Johns Hopkins University (Tianzi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from -1 if you need to start from data download
stop_stage=5
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

preprocess_config=conf/preprocess.json
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=20000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

#  mdm8 - multiple distant microphones
#  this scipt only support AMI-Array1 mutichannel data
mic=mdm8

# exp tag
tag="" # tag for managing experiments

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,
case $(hostname -d) in
    clsp.jhu.edu) AMI_DIR=/export/corpora4/ami/amicorpus ;; # JHU,
esac

train_set=${mic}_train
train_dev=${mic}_dev
recog_set="${mic}_dev ${mic}_eval"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    if [ -d ${AMI_DIR} ] && ! touch ${AMI_DIR}/.foo 2>/dev/null; then
	echo "$0: directory $AMI_DIR seems to exist and not be owned by you."
	echo " ... Assuming the data does not need to be downloaded.  Please use --stage 0 or more."
	exit 1
    fi
    if [ -e data/local/downloads/wget_${mic}.sh ]; then
	echo "data/local/downloads/wget_$mic.sh already exists, better quit than re-download... (use --stage N)"
	exit 1
    fi
    local/ami_download.sh ${mic} ${AMI_DIR}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    # common data prep
    if [ ! -d data/local/downloads ]; then
	local/ami_text_prep.sh data/local/downloads
    fi

    local/ami_mdm_multich_data_prep.sh ${AMI_DIR} mdm8_tmp
   
    # speed perturbation based data augmentation.
    utils/perturb_data_dir_speed.sh 0.9 data/mdm8_tmp/train_orig data/mdm8_tmp1
    utils/perturb_data_dir_speed.sh 1.0 data/mdm8_tmp/train_orig data/mdm8_tmp2
    utils/perturb_data_dir_speed.sh 1.1 data/mdm8_tmp/train_orig data/mdm8_tmp3
    
    utils/combine_data.sh --extra-files utt2uniq data/mdm8_multich/train_orig data/mdm8_tmp1 data/mdm8_tmp2 data/mdm8_tmp1
    
    rm -r data/mdm8_tmp data/mdm8_tmp1 data/mdm8_tmp2 data/mdm8_tmp3
    
    local/ami_mdm_multich_scoring_data_prep.sh ${AMI_DIR} mdm8_multich dev
    local/ami_mdm_multich_scoring_data_prep.sh ${AMI_DIR} mdm8_multich eval

    for dset in train dev eval; do
	# change the original AMI data structure in the Kaldi recipe to the following
	utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 data/mdm8_multich/${dset}_orig data/${mic}_${dset}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ## But you can utilize Kaldi recipes in most cases
    echo "stage 1: Dump wav files into a HDF5 file"

    for setname in ${train_set} ${recog_set}; do
	dump_pcm.sh --nj 32 --cmd "${train_cmd}" --filetype "sound.hdf5" --format flac --write-utt2num-frames true data/${setname}
	local/ami_deprecate_short_utts.sh data/${setname} 0.1 16000
	utils/fix_data_dir.sh data/${setname}
    done    
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt
if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms} || true
    cat ${nlsyms}
    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
	| sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    # "--category" is just a mark to be used excludely for minibatch creation.
    # i.e. A minibatch always consists of the samples in either one of these categories.
    echo "make json files"
    for setname in ${train_set} ${recog_set}; do
        data2json.sh --cmd "${train_cmd}" \
		     --nj 30 \
		     --category "multichannel" \
		     --preprocess-conf ${preprocess_config} \
		     --filetype sound.hdf5 \
		     --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
		     --out data/${setname}/data.json data/${setname} ${dict}
    done

fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi

lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train_trans.txt
        cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
	cat ${lmdatadir}/train_trans.txt > ${lmdatadir}/train.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        cat ${lmdatadir}/train_trans.txt > ${lmdatadir}/train.txt
    fi

    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. single gpu will be used."
    fi
    
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
		lm_train.py \
		--config ${lm_config} \
		--ngpu ${ngpu} \
		--backend ${backend} \
		--verbose 1 \
		--outdir ${lmexpdir} \
		--tensorboard-dir tensorboard/${lmexpname} \
		--train-label ${lmdatadir}/train.txt \
		--valid-label ${lmdatadir}/valid.txt \
		--resume ${lm_resume} \
		--dict ${lmdict}    
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training: expdir=${expdir}"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
		asr_train.py \
		--config ${train_config} \
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
		--train-json data/${train_set}/data.json \
		--valid-json data/${train_dev}/data.json  \
		--preprocess-conf ${preprocess_config} 
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32
    pids=() # initialize pids
    for rtask in ${recog_set}; do
	(
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
            if [ ${use_wordlm} = true ]; then
		recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
            else
		recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
            fi

            # split data
            splitjson.py --parts ${nj} data/${rtask}/data.json

            #### use CPU for decoding
            ngpu=0
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
			  asr_recog.py \
			  --config ${decode_config} \
			  --ngpu ${ngpu} \
			  --backend ${backend} \
			  --debugmode ${debugmode} \
			  --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
			  --result-label ${expdir}/${decode_dir}/data.JOB.json \
			  --model ${expdir}/results/${recog_model}  \
			  ${recog_opts}
            score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
	) &
	pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Decoding successfully finished"
    
fi
