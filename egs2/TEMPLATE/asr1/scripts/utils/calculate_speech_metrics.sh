#!/usr/bin/env bash

# 2021 @wangyou-zhang
# Copied from ./scripts/utils/perturb_data_dir_speed.sh
# Modified for calculating speech related metrics

# Copyright 2021  Shanghai Jiao Tong University (author: Wangyou Zhang)
# Apache 2.0

# This script calculates the specified metric between audios in <ref_scp> and <enh_scp>,
# and store it in the specified output file.


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

export LC_ALL=C

ref_channel=0    # reference channel
nj=32            # The number of parallel jobs in inference
python=python3   # Specify python to execute espnet commands
id_prefix="enh-" # prefix of utt ids to remove in <ref_scp> and <enh_scp>
frame_size=512   # STFT frame size in samples
frame_hop=256    # STFT frame hop in samples

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh


help_message=$(cat << EOF
Usage: calculate_speech_metrics.sh <ref_scp> <enh_scp> <metric> <outfile>
e.g.:
    $0 spk1.scp enh.scp SNR snr.scp

Arguments:
    <ref_scp>: scp file containing the path to audios that will be used as reference signals when calculating the metrics
    <enh_scp>: scp file containing the path to audios that will be used as estimated signals when calculating the metrics
    <metric>: must be one of the following:
        "STOI": short-time objective intelligibility
        "ESTOI": extended short-time objective intelligibility
        "SNR": signal-to-noise ratio
        "SI-SNR": scale-invariant signal-to-noise ratio
        "SDR": signal-to-distortion ratio
        "SAR": signal-to-artifact ratio
        "SIR": signal-to-interference ratio
        "framewise-SNR": frame-level SNR
    <outfile>: the scp file to store the calculated metric

Optional:
    --ref_channel: Reference channel of the reference speech will be used if the enhanced speech is single-channel and reference speech is multi-channel (default=${ref_channel})
    --nj: The number of parallel jobs in inference (default=${nj})
    --python: specify python to execute espnet commands (default=${python})
    --id_prefix: prefix of utt ids to remove in <ref_scp> and <enh_scp> (default=${id_prefix})
    --frame_size: STFT frame size in samples, for calculating framewise-* metrics (default=${frame_size})
    --frame_hop: STFT frame hop in samples, for calculating framewise-* metrics (default=${frame_hop})
EOF
)

log "$0 $*"

if [[ $# != 4 ]]; then
    log "${help_message}"
    exit 2
fi

ref_scp=$1
enh_scp=$2
metric=$3
outfile=$4

tmpdir=$(mktemp -d speech_metrics-XXXX)
chmod 755 "${tmpdir}"


# 1. Split the key file
sed -e "s/${id_prefix}//g" ${ref_scp} > "${tmpdir}/ref.scp"
sed -e "s/${id_prefix}//g" ${enh_scp} > "${tmpdir}/enh.scp"
key_file="${tmpdir}/ref.scp"
split_scps=""
_nj=$(min "${nj}" "$(<${key_file} wc -l)")
for n in $(seq "${_nj}"); do
    split_scps+=" ${tmpdir}/keys.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${key_file}" ${split_scps}

# 2. Submit scoring jobs
log "Scoring started... log: '${tmpdir}/enh_metric.*.log'"
# shellcheck disable=SC2086,SC2154
${decode_cmd} JOB=1:"${_nj}" "${tmpdir}"/enh_metric.JOB.log \
    ${python} pyscripts/utils/calculate_speech_metrics.py \
        --key_file "${tmpdir}"/keys.JOB.scp \
        --output_dir "${tmpdir}"/output.JOB \
        --ref_scp "${tmpdir}/ref.scp" \
        --inf_scp "${tmpdir}/enh.scp" \
        --metrics ${metric} \
        --ref_channel ${ref_channel} \
        --frame_size ${frame_size} \
        --frame_hop ${frame_hop}

for i in $(seq "${_nj}"); do
    cat "${tmpdir}/output.${i}/${metric}_spk1"
done | LC_ALL=C sort -k1 > "${outfile}"

rm -r "${tmpdir}"
log "$0: calculated metric is in ${outfile}"
