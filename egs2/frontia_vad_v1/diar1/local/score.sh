#!/usr/bin/env bash
# VAD scoring of diar.sh stage 7 for this recipe (replaces its default scoring).
#
#   local/score.sh [options] <diar_exp> <data_feats> <set> [<set> ...]
#
# md-eval over each whole recording: a UEM from 0 to the end of every recording
# (from utt2num_samples), so false alarms before the first and after the last
# speech count; without it md-eval scores only from the first to the last
# reference segment. Every (set, threshold, median) is one score_der.sh call; the
# calls are dealt out to --nj jobs of ${decode_cmd}. Then local/show_vad_result.py
# reports every set at the setting tuned on --tune_set (RESULTS_vad.md).
set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

nj=32
thresholds="0.3 0.4 0.5 0.6 0.7 0.8 0.9"
medians="1 11"
collar=0.0
fs=16000          # Hz
frame_shift=128   # samples
tune_set=ami_mh_dev

. utils/parse_options.sh
. ./cmd.sh

if [ $# -lt 3 ]; then
    echo "Usage: $0 [options] <diar_exp> <data_feats> <set> [<set> ...]"
    exit 1
fi
diar_exp=$1
data_feats=$2
shift 2

logdir=${diar_exp}/scoring_logdir
mkdir -p "${logdir}"
rm -f "${logdir}"/score.*.sh
n=0
for dset in "$@"; do
    dir=${diar_exp}/diarized_${dset}/scoring
    mkdir -p "${dir}"
    # <recording> <channel> <start> <end> in seconds: the whole recording
    awk -v fs="${fs}" '{ printf "%s 1 0 %.6f\n", $1, $2 / fs }' \
        "${data_feats}/${dset}/utt2num_samples" > "${dir}/uem"
    for med in ${medians}; do
        for th in ${thresholds}; do
            echo "scripts/utils/score_der.sh --collar ${collar} --fs ${fs}" \
                "--frame_shift ${frame_shift} --thresholds ${th} --medians ${med}" \
                "--uem ${dir}/uem ${dir} ${diar_exp}/diarized_${dset}/diarize.scp" \
                "${data_feats}/${dset}/rttm" >> "${logdir}/score.$((n % nj + 1)).sh"
            n=$((n + 1))
        done
    done
done
jobs=$((n < nj ? n : nj))
log "${n} scorings in ${jobs} jobs, log: ${logdir}/score.*.log"
# shellcheck disable=SC2154
${decode_cmd} JOB=1:"${jobs}" "${logdir}"/score.JOB.log bash -e "${logdir}"/score.JOB.sh

if [ -d "${diar_exp}/diarized_${tune_set}/scoring" ]; then
    python3 local/show_vad_result.py "${diar_exp}" --tune_set "${tune_set}"
else
    log "No ${tune_set} scored: no RESULTS_vad.md"
fi
