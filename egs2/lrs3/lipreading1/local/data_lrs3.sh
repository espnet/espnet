#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh
. ./path.sh
. ./cmd.sh

cmd=run.pl
nj=1

echo ${LRS3}

log_dir=./local/python_debug_log

$cmd JOB=1:$nj ${log_dir}_${nj}.log python ./local/data_prep.py --train_val_path ${LRS3}/trainval --test_path ${LRS3}/test 

# output=(python ./local/data_prep.py --train_val_path ${LRS3}/trainval --test_path ${LRS3}/test)
# echo 4[]