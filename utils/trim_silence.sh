#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

fs=16000
win_length=1024
shift_length=256
threshold=60
min_silence=0.01
normalize=16
cmd=run.pl
nj=32

help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> <log-dir>
e.g.: $0 data/train exp/trim_silence/train
Options:
  --fs <fs>                      # sampling frequency (default=16000)
  --win_length <win_length>      # window length in point (default=1024)
  --shift_length <shift_length>  # shift length in point (default=256)
  --threshold <threshold>        # power threshold in db (default=60)
  --min_silence <sec>            # minimum silence lenght in sec (default=0.01)
  --normalize <bit>              # audio bit (default=16)
  --cmd <cmd>                    # how to run jobs (default=run.pl)
  --nj <nj>                      # number of parallel jobs (default=32)
EOF
)

. utils/parse_options.sh || exit 1;

if [ ! $# -eq 2 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail
data=$1
logdir=$2

tmpdir=$(mktemp -d ${data}/tmp-XXXX)
split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${tmpdir}/wav.${n}.scp"
done
utils/split_scp.pl ${data}/wav.scp ${split_scps} || exit 1;

# make segments file describing start and end time
${cmd} JOB=1:${nj} ${logdir}/trim_silence.JOB.log \
    MPLBACKEND=Agg trim_silence.py \
        --fs ${fs} \
        --win_length ${win_length} \
        --shift_length ${shift_length} \
        --threshold ${threshold} \
        --min_silence ${min_silence} \
        --normalize ${normalize} \
        --figdir ${logdir}/figs \
        scp:${tmpdir}/wav.JOB.scp \
        ${tmpdir}/segments.JOB

# concatenate segments
for n in $(seq ${nj}); do
    cat ${tmpdir}/segments.${n} || exit 1;
done > ${data}/segments || exit 1
rm -rf ${tmpdir}

# check
utils/validate_data_dir.sh --no-feats ${data}
echo "Successfully trimed silence part."
