#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=0
stage_last=1000000

# Data settings 
do_delta=false # true when using CNN
do_cvn=true
do_cvn=false #true

dumpdir=dump.8k   # directory to dump full features
dumpdir_name=delta${do_delta}cvn$do_cvn # 

# Input/output dirs
dir=exp/dnn_24fbank8k.bn3L #3L
feadir_bn=data-24fbank8k.bn3L
ali=exp/tri5b_MultRDTv1_short_ali
graph_src=exp/tri5b_MultRDTv1_short/graph.lang_test.wrd2grp.traindev.o3g.kn
lang_train=data/lang.wrd2grp
njdec=30
njfea=10

# Train test sets
train_set=train
train_dev=dev
recog_set="eval_101"

. utils/parse_options.sh || exit 1;




fbankdir=fbank24.8k
if [ $stage -le 1 ] && [ $stage_last -ge 1 ]; then

    # Generate and dump features
    for x in ${train_set} ${train_dev} ${recog_set}; do
	copy_data_dir.sh data/$x data/$x.8k; rm -f data/$x.8k/{feats.scp,cmvn*}
	cp  data/$x/wav.scp.bak  data/$x.8k/wav.scp
	cp  data/$x/wav.scp.bak  data/$x.8k/wav.scp.bak
	local/makeanddump_fea.8k.sh \
	    --data_in data/$x.8k \
	    --data_fea ${fbankdir}/$x \
	    --data_dmp ${dumpdir}/$x/${dumpdir_name}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/${dumpdir_name}
feat_dt_dir=${dumpdir}/${train_dev}/${dumpdir_name}
feat_et_dir=${dumpdir}/${recog_set}/${dumpdir_name}

for i in ${train_set} ${train_dev} ${recog_set}; do
    d_in=data/${i}
    d_out=${dumpdir}/${i}/${dumpdir_name}
    if [ ! -s ${d_out}/cmvn.scp ]; then
	cp ${d_in}/{segments,spk2utt,utt2spk,text,wav.scp} ${d_out}
	steps/compute_cmvn_stats.sh ${d_out} ${d_out}/log ${d_out}/data || exit 1;
    fi
done


# Train the bottleneck network,
if [ $stage -le 2 ] && [ $stage_last -ge 2 ]; then
    $cuda_cmd $dir/log/train_nnet.log \
	steps/nnet/train.sh --hid-layers 3 --hid-dim 1500 --bn-dim 40 \
	--cmvn-opts "--norm-means=false --norm-vars=false" --feat-type traps \
	--splice 5 --traps-dct-basis 6 --learn-rate 0.008 \
	${feat_tr_dir} ${feat_dt_dir} ${lang_train} $ali $ali $dir || exit 1
    # Decode recog_set
    steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
	--scoring-opts "--min-lmwt 10 --max-lmwt 20" --num-threads 2 --parallel-opts "-pe smp 2" --max-mem 150000000 \
	$graph_src ${feat_et_dir} $dir/decode_${recog_set}.$(basename $graph_src) || exit 1
fi

# Store the bottleneck features,
if [ $stage -le 3 ] && [ $stage_last -ge 3 ]; then
    
    for i in ${train_set} ${train_dev} ${recog_set}; do
	feadir_in=${dumpdir}/${i}/${dumpdir_name}
	feadir_out=${feadir_bn}/${i}
	
	if [ ! -e ${feadir_out}/feats.scp ]; then
 	    steps/nnet/make_bn_feats.sh --nj $njfea --cmd "$train_cmd" ${feadir_out} ${feadir_in} $dir ${feadir_out}/log ${feadir_out}/data || exit 1
	    steps/compute_cmvn_stats.sh ${feadir_out} ${feadir_out}/log ${feadir_out}/data || exit 1;
	fi
    done
fi
echo "Finished"
