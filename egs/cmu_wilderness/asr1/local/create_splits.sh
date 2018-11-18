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
for l in `cat ${trainset} | tr "\n" " "`; do
  train="${dir}/${l}_train ${train}"
done
train=${train%% }

# Dev
dev=""
for l in `cat ${devsetloop} | tr "\n" " "`; do
  dev="${dir}/${l}_dev ${dev}"
done
dev=${dev%% }

# Eval
eval=""
for l in `cat ${evalsetloop} | tr "\n" " "`; do
  eval="${dir}/${l}_eval ${eval}"
done
eval=${eval%% }

# Combine
for d in train dev eval; do
  ./utils/combine_data.sh --extra-files text.phn data/${d} ${!d}
done
