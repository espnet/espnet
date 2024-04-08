#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a task of 42 language-indepent ASR

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
lid=""        # add language id, using "" will not add it
dumpdir=dump${lid}   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10

# exp tag
tag="" # tag for managing experiments.

# combine more corpora for training by adding their espnet-format data dir
aishell="-"
aurora4="-"
babel="-"
chime4="-"
commonvoice="-"
csj="-"
fisher_callhome_spanish="-"
fisher_swbd="-"
hkust="-"
voxforge="-"
wsj="-"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_li42
train_dev=dev_li42
recog_set="dt_*_commonvoice dt_*_babel dt_*_wsj dt_*_aurora4 dt_*_csj dt_*_fisher_swbd dt_*_hkust dt_*_aishell dt_*_fisher_callhome_spanish dt_*_chime4 dt_*_voxforge et_*_commonvoice et_*_babel et_*_wsj et_*_csj et_*_fisher_swbd et_*_hkust et_*_aishell et_*_fisher_callhome_spanish et_*_chime4 et_*_voxforge" # can either use patterns or full names here.

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    lang_code=zh
    if [ -e ${aishell} ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_aishell ${aishell}/train data/tr_${lang_code}_aishell
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_aishell ${aishell}/dev   data/dt_${lang_code}_aishell
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_aishell ${aishell}/test  data/et_${lang_code}_aishell
    else
	echo "no aishell data directory found"
	echo "cd ../../aishell/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi
    if [ -e ${hkust} ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_hkust ${hkust}/train_nodup_sp data/tr_${lang_code}_hkust
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_hkust ${hkust}/train_dev      data/dt_${lang_code}_hkust
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_hkust ${hkust}/dev            data/et_${lang_code}_hkust
    else
	echo "no hkust data directory found"
	echo "cd ../../hkust/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi

    # CSJ Japanese
    lang_code=ja
    if [ -e ${csj} ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_csj ${csj}/train_nodup data/tr_${lang_code}_csj
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_csj ${csj}/train_dev   data/dt_${lang_code}_csj
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_csj ${csj}/eval1       data/et_${lang_code}_1_csj
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_csj ${csj}/eval2       data/et_${lang_code}_2_csj
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_csj ${csj}/eval3       data/et_${lang_code}_3_csj
	# 1) change wide to narrow chars
	# 2) lower to upper chars
	for x in data/*_"${lang_code}"*; do
            utils/copy_data_dir.sh ${x} ${x}_org
            < ${x}_org/text nkf -Z |\
		awk '{for(i=2;i<=NF;++i){$i = toupper($i)} print}' > ${x}/text
            rm -fr ${x}_org
	done
    else
	echo "no csj data directory found"
	echo "cd ../../csj/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi

    lang_code=en
    if [ -e ${aurora4} ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_aurora4 ${aurora4}/train_si84_multi data/tr_${lang_code}_aurora4
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_aurora4 ${aurora4}/dev_0330         data/dt_${lang_code}_aurora4
    else
	echo "no aurora4 data directory found"
	echo "cd ../../aurora4/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi
    if [ -e ${chime4} ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_chime4 ${chime4}/tr05_multi_noisy              data/tr_${lang_code}_chime4
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_chime4 ${chime4}/dt05_multi_isolated_1ch_track data/dt_${lang_code}_chime4
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_chime4 ${chime4}/et05_real_isolated_1ch_track  data/et_${lang_code}_real_chime4
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_chime4 ${chime4}/et05_simu_isolated_1ch_track  data/et_${lang_code}_simu_chime4
    else
	echo "no chime4 data directory found"
	echo "cd ../../chime4/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi
    if [ -e ${fisher_swbd} ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_fisher_swbd ${fisher_swbd}/train     data/tr_${lang_code}_fisher_swbd
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_fisher_swbd ${fisher_swbd}/rt03_trim data/dt_${lang_code}_fisher_swbd
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_fisher_swbd ${fisher_swbd}/eval2000  data/et_${lang_code}_fisher_swbd
    else
	echo "no fisher_swbd data directory found"
	echo "cd ../../fisher_swbd/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi
    if [ -e ${wsj} ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_wsj ${wsj}/train_si284 data/tr_${lang_code}_wsj
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_wsj ${wsj}/test_dev93  data/dt_${lang_code}_wsj
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_wsj ${wsj}/test_eval92 data/et_${lang_code}_wsj
    else
	echo "no wsj data directory found"
	echo "cd ../../wsj/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi

    lang_code=es
    if [ -e ${fisher_callhome_spanish} ]; then
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_fisher_callhome_spanish ${fisher_callhome_spanish}/train         data/tr_${lang_code}_fisher_callhome_spanish
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_fisher_callhome_spanish ${fisher_callhome_spanish}/dev           data/dt_${lang_code}_fisher_callhome_spanish
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_fisher_callhome_spanish ${fisher_callhome_spanish}/test          data/et_${lang_code}_1_fisher_callhome_spanish
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_fisher_callhome_spanish ${fisher_callhome_spanish}/callhome_dev  data/et_${lang_code}_2_fisher_callhome_spanish
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_fisher_callhome_spanish ${fisher_callhome_spanish}/callhome_test data/et_${lang_code}_3_fisher_callhome_spanish
    else
	echo "no fisher_callhome_spanish data directory found"
	echo "cd ../../fisher_callhome_spanish/asr1/; ./run.sh --stop_stage 2; cd -"
	exit 1;
    fi

    # Babel
    for x in 101-cantonese 102-assamese 103-bengali 104-pashto 105-turkish 106-tagalog 107-vietnamese \
	     201-haitian 202-swahili 203-lao 204-tamil 205-kurmanji 206-zulu 207-tokpisin \
	     301-cebuano 302-kazakh 303-telugu 304-lithuanian 305-guarani 306-igbo 307-amharic \
	     401-mongolian 402-javanese 403-dholuo 404-georgian; do
        langid=$(echo ${x} | cut -f 1 -d"-")
	lang_code=$(echo ${x} | cut -f 2 -d"-")
	if [ ! -d "${babel}/asr1_${lang_code}/data" ]; then
	    echo "run ../../babel/asr1/local/run_all_stage1.sh first"
	    exit 1
	fi
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_babel ${babel}/asr1_${lang_code}/data/train          data/tr_${lang_code}_babel
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_babel ${babel}/asr1_${lang_code}/data/dev            data/dt_${lang_code}_babel
	utils/copy_data_dir.sh --utt-suffix -${lang_code}_babel ${babel}/asr1_${lang_code}/data/eval_${langid} data/et_${lang_code}_babel
    done
    # remove space in Cantonese
    for x in data/*_cantonese*; do
	cp ${x}/text ${x}/text.org
	paste -d " " <(cut -f 1 -d" " ${x}/text.org) <(cut -f 2- -d" " ${x}/text.org | tr -d " ") \
	      > ${x}/text
	rm ${x}/text.org
    done

    # Voxforge
    for lang_code in de es fr it nl pt ru; do
	if [ -e ${voxforge}/tr_${lang_code} ]; then
            utils/copy_data_dir.sh --utt-suffix -${lang_code}_voxforge ${voxforge}/tr_${lang_code} data/tr_${lang_code}_voxforge
            utils/copy_data_dir.sh --utt-suffix -${lang_code}_voxforge ${voxforge}/dt_${lang_code} data/dt_${lang_code}_voxforge
            utils/copy_data_dir.sh --utt-suffix -${lang_code}_voxforge ${voxforge}/et_${lang_code} data/et_${lang_code}_voxforge
	else
	    echo "no voxforge ${lang_code} data directory found"
	    echo "cd ../../voxforge/asr1/; ./run.sh --stop_stage 2 --lang ${lang_code}; cd -"
	    exit 1;
	fi
    done

    # commonvoice
    if [ -e ${commonvoice} ]; then
        for lang_code in en de fr cy tt kab ca zh_TW it fa eu es ru; do
            utils/copy_data_dir.sh --utt-suffix -${lang_code}_commonvoice ${commonvoice}/valid_train_${lang_code} data/tr_${lang_code}_commonvoice
            utils/copy_data_dir.sh --utt-suffix -${lang_code}_commonvoice ${commonvoice}/valid_dev_${lang_code} data/dt_${lang_code}_commonvoice
            utils/copy_data_dir.sh --utt-suffix -${lang_code}_commonvoice ${commonvoice}/valid_test_${lang_code} data/et_${lang_code}_commonvoice
        done
    else
        echo "no commonvoice data directory found"
        echo "cd ../../commonvoice/asr1/; ./run.sh --stop_stage 2; cd -"
        exit 1;
    fi
fi

# Set real recog_set
recog_set_new=""
for rtask_pattern in ${recog_set}; do
    rtasks=$(find data/ -name "${rtask_pattern}" | sed 's!^.*/!!')
    recog_set_new="${recog_set_new} ${rtasks}"
done
recog_set=${recog_set_new}

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # for some reason fix_data_dir.sh is not working after the combination
    # and I decided to perform it before the combination and skip it after the combination
    for x in data/tr_* data/dt_* data/et_* ; do
	utils/fix_data_dir.sh ${x}
	# remove utt having more than 2500 frames or less than 10 frames or
	# remove utt having more than 250 characters or 0 characters
	remove_longshortdata.sh --maxframes 2500 --maxchars 250 ${x} ${x}_trim
    done
    utils/combine_data.sh --skip_fix true data/${train_set} data/tr_*_trim
    utils/combine_data.sh --skip_fix true data/${train_dev} data/dt_*_trim

    # normalize the case (lower to upper)
    for x in data/${train_set} data/${train_dev}; do
	cp ${x}/text ${x}/text.org
	paste -d " " \
	      <(cut -f 1 -d" " ${x}/text.org) \
	      <(cut -f 2- -d" " ${x}/text.org | python3 -c 'import sys; print(sys.stdin.read().upper(), end="")') \
	      > ${x}/text
	rm ${x}/text.org
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/li10/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/li10/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

# new data adding language ID tag
if [ "$lid" != "" ]
then
    train_set_org="${train_set}"
    train_dev_org="${train_dev}"
    train_set=${train_set}_lid
    train_dev=${train_dev}_lid
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/${train_set}_non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # utils/copy_data_dir.sh --validate_opts --no-spk-sort data/${train_set_org} data/${train_set}
    # utils/copy_data_dir.sh --validate_opts --no-spk-sort data/${train_dev_org} data/${train_dev}
    # add lid
    if [ "$lid" != "" ]
    then
        paste -d " " \
	  <(cut -f 1 -d" " data/${train_set_org}/text) \
	  <(cut -f 1 -d" " data/${train_set_org}/text | sed -e "s/.*\-\(.*\)_.*/\1/" | sed -e "s/_[^TW]\+//" | sed -e "s/^/\[/" -e "s/$/\]/") \
	  <(cut -f 2- -d" " data/${train_set_org}/text) | sed -e "s/\([^[]*\[[^]]*\]\)\s\(.*\)/\1\2/" \
	  > data/${train_set}/text
        paste -d " " \
	  <(cut -f 1 -d" " data/${train_dev_org}/text) \
	  <(cut -f 1 -d" " data/${train_dev_org}/text | sed -e "s/.*\-\(.*\)_.*/\1/" | sed -e "s/_[^TW]\+//" | sed -e "s/^/\[/" -e "s/$/\]/") \
	  <(cut -f 2- -d" " data/${train_dev_org}/text) | sed -e "s/\([^[]*\[[^]]*\]\)\s\(.*\)/\1\2/" \
	  > data/${train_dev}/text
    fi

    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --cmd "$train_cmd" --nj 80 --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --cmd "$train_cmd" --nj 32 --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --cmd "$train_cmd" --nj 32 --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
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
mkdir -p ${expdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
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
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    nj=28
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
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use GPU for decoding
        ngpu=1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --api v2 \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        score_sclite.sh --nlsyms ${nlsyms} --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
