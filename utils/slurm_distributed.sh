#!/bin/bash

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

gpu=0
node=0

. utils/parse_options.sh || exit 1;

. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

nprocs=$node #$((${gpu} * ${node}))

${cuda_cmd} --gpu ${gpu} --num_procs ${nprocs} --num_nodes ${node} $1 \
            srun -n${nprocs} -N${node} --gres gpu:${gpu} ${@:2}
