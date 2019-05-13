#!/bin/bash

data=$1

if [ ! $# -eq 1 ]; then
   echo "Usage: $0 [options] <data-dir>";
   echo "e.g.: $0 data/train"
   echo "  --rirdir RIRS_NOISES directory path (default=/fsws1/share/database/RIRS_NOISES)."
   echo "  --sampling_rate sampling rate (default=16000)."
   echo "  --num_replications number of replications (default=5)."
   exit 1;
fi

. ./path.sh

rirdir=/fsws1/share/database/RIRS_NOISES
sampling_rate=16000
num_replications=5

. utils/parse_options.sh

# Make a version with reverberated speech
rvb_opts=()
rvb_opts+=(--rir-set-parameters "0.5, ${rirdir}/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, ${rirdir}/simulated_rirs/mediumroom/rir_list")

# Make reveration data
steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications ${num_replications} \
    --source-sampling-rate ${sampling_rate} \
    --prefix rvb \
    ${data} $(basename $data)_reverb

# Combine all directories
utils/combine_data.sh $(basename $data)_aug ${data} $(basename $data)_reverb