#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Trim silence and generate segments file.

SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

fs=16000
win_length=1024
shift_length=256
threshold=35
min_silence=0.01
normalize=16
cmd=run.pl
nj=32

help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> <log-dir>
e.g.: $0 data/train exp/trim_silence/train
Options:
  --fs <fs>                      # Sampling frequency (default="${fs}").
  --win_length <win_length>      # Window length in point (default="${win_length}").
  --shift_length <shift_length>  # Shift length in point (default="${shift_length}").
  --threshold <threshold>        # Power threshold in db (default="${threshold}").
  --min_silence <sec>            # Minimum silence length in sec (default="${min_silence}").
  --normalize <bit>              # Audio bit (default="${normalize}").
  --cmd <cmd>                    # How to run jobs (default="${cmd}").
  --nj <nj>                      # Number of parallel jobs (default="${nj}").
EOF
)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

if [ ! $# -eq 2 ]; then
    log "${help_message}"
    log "Error: invalid command line arguments"
    exit 1;
fi

set -euo pipefail

# shellcheck disable=SC1091
. ./path.sh

datadir=$1
logdir=$2

[ ! -e "${logdir}" ] && mkdir -p "${logdir}"
tmpdir=$(mktemp -d "${logdir}"/tmp-XXXX)
split_scps=""
for n in $(seq "${nj}"); do
    split_scps="${split_scps} ${tmpdir}/wav.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${datadir}/wav.scp" ${split_scps} || exit 1;

# make segments file describing start and end time
${cmd} JOB=1:"${nj}" "${logdir}/trim_silence.JOB.log" \
    MPLBACKEND=Agg pyscripts/audio/trim_silence.py \
        --fs "${fs}" \
        --win_length "${win_length}" \
        --shift_length "${shift_length}" \
        --threshold "${threshold}" \
        --min_silence "${min_silence}" \
        --normalize "${normalize}" \
        --figdir "${logdir}/figs.JOB" \
        scp:"${tmpdir}/wav.JOB.scp" \
        "${tmpdir}/segments.JOB"

# concatenate segments
for n in $(seq "${nj}"); do
    cat "${tmpdir}/segments.${n}" || exit 1;
done > "${datadir}/segments" || exit 1
rm -rf "${tmpdir}"

# check
utils/fix_data_dir.sh "${datadir}"
log "Successfully finished silence trimming. [elapsed=${SECONDS}s]"
