#!/bin/bash

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./cmd.sh

gpu=0
node=0
uniform=true
cmd=${cuda_cmd}

. utils/parse_options.sh || exit 1;

. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

echo $@

if [ ${node} -eq 0 ]; then
    echo "[INFO]: single node job"
    ${cmd} --gpu ${gpu} $@
    exit 0
fi

if [ ${gpu} -eq 0 ]; then
    echo "[ERROR]: CPU job (--gpu 0) is not supported for multi node job"
    exit 0
fi

if ${uniform}; then
    if [ ${cmd} = "slurm.pl" ]; then
        echo "[INFO]: multi node job with uniform distributed GPUs using SLURM"
        ${cmd} --gpu ${gpu} --num_nodes ${node} $1 srun -N${node} --gres gpu:${gpu} ${@:2}
        exit 0
    fi
else
    echo "[INFO]: multi node job with greedy distributed GPUs"
    world=$((${gpu} * ${node}))
    max_rank=$((${world} - 1))
    for rank in {0..${max_rank}}; do
        ${cmd} --gpu 1 $@ --ngpu 1 --rank ${rank} --world-size ${world} &
    done
    wait
    exit 0
fi

