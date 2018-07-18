#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh


# Data settings 
feakind=bn3L

# general configuration
extra_train_opts=
extra_eval_opts=


# Train test sets
train_set=train
train_dev=dev
recog_set=

. utils/parse_options.sh || exit 1;


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $stage -le 0 ] && [ $stage_last -ge 0 ]; then
  echo "stage 0: Setting up individual languages"
  # It will fill directories for all $lang_ids: data/${lang_id}/data 
  #     by train_${lang_id} dev_${lang_id} 
  # and combine them together into main data/{train,dev}
# ./local/setup_languages.sh --langs "${lang_id}" --recog "${lang_id}"
fi

fbankdir=data-$feakind
lang=data/lang_1char

if [ ${stage} -le 2 ] && [ ${stage_last} -ge 2 ]; then
    # --- Prepare $lang/train_unit.dct
    if [ ! -d $lang ]; then
	echo "Creating dictionary: into  ${lang}"
	./local/make_dct.sh \
	    --lang $lang    \
	    --data "data/${train_set} data/${train_dev}"
    fi

    for x in ${train_set} ${train_dev} ${recog_set}; do
	./local/make_json.sh \
	    --lang $lang                    \
	    --data_in $fbankdir/$x --data $fbankdir/$x
    done
fi


if [ ${stage} -le 3 ]; then
    # NN train /eval
    ./run/train_espnet.sh \
	--data_train  $fbankdir/${train_set} \
	--data_dev    $fbankdir/${train_dev} \
	--data_eval   $fbankdir/${recog_set} \
	--tagflag _${feakind} \
	--extra_train_opts "$extra_train_opts" \
	--extra_eval_opts  "$extra_eval_opts"
fi
