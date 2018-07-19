#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a task of 10 language-indepent ASR used in
# S. Watanabe et al, "Language independent end-to-end architecture for
# joint language identification and speech recognition," Proc. ASRU'17, pp. 265--269 (2017)

. ./path.sh
. ./cmd.sh


stage=0
# -- data
data_train=dump/train/deltafalsecvntrue
data_dev=dump/dev/deltafalsecvntrue
data_eval=dump/eval_202/deltafalsecvntrue

# dct
lang=data/lang_1char # dir where units and non lang symbols are expected

# -- MultNN info
multnn_dir=/mnt/matylda6/baskar/espnet_forked/egs/babel/asr1/exp/tr_babel10_blstmp_e5_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs50_mli800_mlo150
multnn_resume=XXXXX  #no implemented yet but opt has to be set #   $multnn_dir/results/snapshot_iter_12847
expname=multnn_karthick_v0

train_conf=conf/espnet.base.conf  # file where $train_opt and $tag variable is saved
eval_conf=conf/espnet.base.conf   # file where $eval_opt is defined


. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


dict=$lang/train_units.txt
nlsyms=$lang/non_lang_syms.txt


## Adaptation of the language independant (LI) trained model towards target languages

mexpdir=exp/${expname}.adadelta.AttCtcOut
m2expdir=${mexpdir}_it2

if [ ${stage} -le 1 ]; then
    echo "stage 1: NN language transfer"

    elayers=$(source $train_conf; echo $elayers)
    extra_train_opts="--modify-output true \
        --resume $multnn_resume \
        --epochs 10 \
        --adapt yes \
        --adapt-layer-names AttCtcOut \
        --freeze yes \
        --noencs-freeze $((elayers * 2))
"

    if [ ! -f ${mexpdir}/model.loss.best ]; then
#	mkdir -p $mexpdir/results; cp $multnn_dir/results/{model.acc.best,model.conf} $mexpdir/results
	mkdir -p $mexpdir/results; cp $multnn_dir/results/model.acc.best $mexpdir/results
	
	./run/train_espnet.sh \
	    --train_conf $train_conf \
	    --eval_conf $eval_conf \
	    --expdir ${mexpdir}  \
	    --extra_train_opts "${extra_train_opts}" \
	    --data_train $data_train \
	    --data_dev   $data_dev \
	    --data_eval  $data_eval
    fi
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2: NN fine tunning"

    if [ ! -f ${m2expdir}/model.loss.best ]; then
	mkdir -p $m2expdir/results; cp $mexpdir/results/{model.acc.best,model.conf} $m2expdir/results

	epochs=15
	extra_train_opts="   --opt sgd \
            --lr 1e-2 \
            --epochs ${epochs} \
            --adapt yes"
	
	./run/train_espnet.sh \
	    --train_conf $train_conf \
	    --eval_conf $eval_conf \
	    --expdir ${m2expdir}  \
	    --extra_train_opts "${extra_train_opts}" \
	    --data_train $data_train \
	    --data_dev   $data_dev \
	    --data_eval  $data_eval
    fi
fi
