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

ref_channel=0         # reference channel
nj=32       # The number of parallel jobs in inference
python=python3        # Specify python to execute espnet commands
id_prefix="enh-"      # prefix of utt ids to remove in <ref_scp> and <enh_scp>

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh


if [[ $# != 4 ]]; then
    echo 'Usage: calculate_speech_metrics.sh <ref_scp> <enh_scp> <metric> <outfile>'
    echo "e.g.:"
    echo " $0 spk1.scp enh.scp SNR snr.scp"
    echo -e "\nCurrent supported metrics:"
    echo "  * STOI: short-time objective intelligibility"
    echo "  * ESTOI: extended short-time objective intelligibility"
    echo "  * SNR: signal-to-noise ratio"
    echo "  * SI-SNR: scale-invariant signal-to-noise ratio"
    echo "  * SDR: signal-to-distortion ratio"
    echo "  * SAR: signal-to-artifact ratio"
    echo "  * SIR: signal-to-interference ratio"
    echo -e "\nOptional:"
    echo "  --ref_channel: Reference channel of the reference speech will be used if the enhanced speech is single-channel and reference speech is multi-channel (default=${ref_channel})"
    echo "  --nj: The number of parallel jobs in inference (default=${nj})"
    echo "  --python: specify python to execute espnet commands (default=${python})"
    echo "  --id_prefix: prefix of utt ids to remove in <ref_scp> and <enh_scp> (default=${id_prefix})"
    exit 1
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
echo "Scoring started... log: '${tmpdir}/enh_metric.*.log'"
# shellcheck disable=SC2086,SC2154
${decode_cmd} JOB=1:"${_nj}" "${tmpdir}"/enh_metric.JOB.log \
    ${python} scripts/utils/calculate_speech_metrics.py \
        --key_file "${tmpdir}"/keys.JOB.scp \
        --output_dir "${tmpdir}"/output.JOB \
        --ref_scp "${tmpdir}/ref.scp" \
        --inf_scp "${tmpdir}/enh.scp" \
        --metrics ${metric} \
        --ref_channel ${ref_channel}

for i in $(seq "${_nj}"); do
    cat "${tmpdir}/output.${i}/${metric}_spk1"
done | LC_ALL=C sort -k1 > "${outfile}"

rm -r "${tmpdir}"
echo "$0: calculated metric is in ${outfile}"
