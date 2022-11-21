#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0 <vad_home> <data_dir>
e.g.
$0 tools/rVADfast data/test/

Generate vad.scp for corresponding data

Options
  --nj <nj>
  --cmd <cmd>
  --out_filename <out_filename>
EOF
)

out_filename=vad.scp
cmd=utils/run.pl
nj=30

ref_channels=
utt2ref_channels=

log "$0 $*"
. utils/parse_options.sh

. ./path.sh  # Setup the environment

vad_home=$1
dir=$2
if [ ! -f "${dir}/wav.scp" ]; then
    log "${help_message}"
    echo "$0: Error: No wav.scp found in: ${dir}/wav.scp"
    exit 1
fi
logdir=${dir}/vad_logs

mkdir -p ${logdir}

rm -f "${dir}/${out_filename}"


nutt=$(<${dir}/wav.scp wc -l)
nj=$((nj<nutt?nj:nutt))

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
done

utils/split_scp.pl "${dir}/wav.scp" ${split_scps}
${cmd} "JOB=1:${nj}" "${logdir}/compute_vad.JOB.log" \
    pyscripts/audio/compute_vad.py \
    "${vad_home}" \
    "${logdir}/wav.JOB.scp" \
    "${logdir}/vad.JOB.scp"

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat "${logdir}/vad.${n}.scp" || exit 1;
done > "${dir}/${out_filename}" || exit 1


log "Successfully finished. [elapsed=${SECONDS}s]"
