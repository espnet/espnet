#!/usr/bin/env bash

# Copyright 2021 Peter Wu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db_root=$1
spk=$2

available_spks=(
    "slt" "clb" "bdl" "rms" "jmk" "awb" "ksp"
)

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <db_root> <spk>"
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

# check speakers
if ! $(echo ${available_spks[*]} | grep -q ${spk}); then
    echo "Specified spk (${spk}) is not available or not supported." >&2
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${db_root}/${spk}.done" ]; then
    mkdir -p "${db_root}"
    cd "${db_root}" || exit 1;
    wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_${spk}_arctic-0.95-release.tar.bz2
    tar xf cmu_us_${spk}*.tar.bz2
    rm cmu_us_${spk}*.tar.bz2
    cd "${cwd}" || exit 1;
    echo "Successfully finished download."
    touch ${db_root}/${spk}.done
else
    echo "Already exists. Skip download."
fi
