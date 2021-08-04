#!/bin/bash -e

# Copyright 2021 Carnegie Mellon University (Peter Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
spk=$2

available_spks=(
    "hin_ab" "tel_ss" "tam_sdr" "kan_plv" "mar_slp" "guj_dp" "ben_rm"
)

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <db_root_dir> <spk>"
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

# check speakers
if ! $(echo ${available_spks[*]} | grep -q ${spk}); then
    echo "Specified spk (${spk}) is not available or not supported." >&2
    exit 1
fi

# download dataset
cwd=`pwd`
if [ ! -e ${db}/${spk}.done ]; then
    mkdir -p ${db}
    cd ${db}
    wget http://festvox.org/h2r_indic/cmu_indic_${spk}.tar.bz2
    tar xf cmu_indic_${spk}.tar.bz2
    rm cmu_indic_${spk}.tar.bz2
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/${spk}.done
else
    echo "Already exists. Skip download."
fi
