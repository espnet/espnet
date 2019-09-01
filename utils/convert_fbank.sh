#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Begin configuration section.
nj=4
fs=22050
fmax=
fmin=
n_fft=1024
n_shift=512
win_length=
n_mels=
iters=64
cmd=run.pl
help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
e.g.: $0 data/train exp/griffin_lim/train wav
Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data
Options:
  --nj <nj>                  # number of parallel jobs
  --fs <fs>                  # sampling rate
  --fmax <fmax>              # maximum frequency
  --fmin <fmin>              # minimum frequency
  --n_fft <n_fft>            # number of FFT points (default=1024)
  --n_shift <n_shift>        # shift size in point (default=256)
  --win_length <win_length>  # window length in point (default=)
  --n_mels <n_mels>          # number of mel basis (default=80)
  --iters <iters>            # number of Griffin-lim iterations (default=64)
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
EOF
)
# End configuration section.

echo "$0 $*"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "${help_message}"
    exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=${data}/log
fi
if [ $# -ge 3 ]; then
  wavdir=$3
else
  wavdir=${data}/data
fi

# use "name" as part of name of the archive.
name=$(basename ${data})

mkdir -p ${wavdir} || exit 1;
mkdir -p ${logdir} || exit 1;

scp=${data}/feats.scp

split_scps=""
for n in $(seq ${nj}); do
    split_scps="$split_scps $logdir/feats.$n.scp"
done

utils/split_scp.pl ${scp} ${split_scps} || exit 1;

${cmd} JOB=1:${nj} ${logdir}/griffin_lim_${name}.JOB.log \
    convert_fbank_to_wav.py \
        --fs ${fs} \
        --fmax ${fmax} \
        --fmin ${fmin} \
        --win_length ${win_length} \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --n_mels ${n_mels} \
        --iters ${iters} \
        scp:${logdir}/feats.JOB.scp \
        ${wavdir}

rm ${logdir}/feats.*.scp 2>/dev/null

echo "Succeeded creating wav for $name"
