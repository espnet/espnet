#!/bin/bash

. ./path.sh
. ./cmd.sh

. ./utils/parse_options.sh

if [ $# -lt 2 ]; then
  echo "Usage: ./local/create_splits.sh <dir> <train> [<dev> [<eval>]]"
  echo "If dev is not specified all dev sets from train languages are used for"
  echo "dev and all eval sets for all train languages are used for eval."
  echo ""
  echo "If only dev is specified then the eval set corresponding to every"
  echo "language listed in dev is used."
  exit 1;
fi

dir=$1
trainset=$2
devset=$3
evalset=$4

# Parse inputs
devsetloop=${trainset}
if [[ $# -ge 3 && -f ${devset} ]]; then
  devsetloop=${devset}
fi

evalsetloop=${devsetloop}
if [[ $# -ge 4 && -f ${evalset} ]]; then
  evalsetloop=${evalset}
fi

# Train
train=""
merge_train_name="train"
for l in `cat ${trainset} | tr "\n" " "`; do
  train="${dir}/${l}_train ${train}"
  merge_train_name="${l}_${merge_train_name}"
done
train=${train%% }

# Dev
dev=""
merge_dev_name="dev"
for l in `cat ${devsetloop} | tr "\n" " "`; do
  dev="${dir}/${l}_dev ${dev}"
  merge_dev_name="${l}_${merge_dev_name}"
done
dev=${dev%% }

# Eval
eval=""
merge_eval_name="eval"
for l in `cat ${evalsetloop} | tr "\n" " "`; do
  eval="${dir}/${l}_eval ${eval}"
  merge_eval_name="${l}_${merge_eval_name}"
done
eval=${eval%% }

trainset_name="`basename ${trainset}`_train"
devset_name="`basename ${devset}`_dev"
evalset_name="`basename ${evalset}`_eval"

# Combine
./utils/combine_data.sh --extra-files text.phn data/${trainset_name} ${train}
./utils/combine_data.sh --extra-files text.phn data/${devset_name} ${dev}
./utils/combine_data.sh --extra-files text.phn data/${evalset_name} ${eval}
