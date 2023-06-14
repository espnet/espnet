#!/usr/bin/env bash

# 2021 @wangyou-zhang
# Copied from ./scripts/utils/perturb_data_dir_speed.sh
# Modified for enhancing a dataset with a pretrained SE model

# Copyright 2021  Shanghai Jiao Tong University (author: Wangyou Zhang)
# Apache 2.0

# This script operates on a directory, such as in data/train/,
# which contains the following files:
#  wav.scp
#  utt2spk
#
# It enhances the speech data with a specified (pretrained) SE model,
# and generates the corresponding files pointing to the enahnced data.


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

audio_dir=            # The output directory for storing enahnced audio files
scp_files=            # Additional scp files to be copied and converted
spk_num=1             # Number of speakers in the input (>1 for separation)
gpu_inference=false   # Whether to perform gpu inference
inference_nj=32       # The number of parallel jobs in inference
fs=16k                # Sampling rate
python=python3        # Specify python to execute espnet commands
id_prefix="enh-"      # prefix for utt ids and spk ids of enhanced samples
enh_args="--normalize_output_wav true"

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

help_message=$(cat << EOF
Usage: enhance_dataset.sh <srcdir> <destdir> <modelfile>
e.g.:
    $0 dump/train_noisy data/train_noisy_enh '/path/to/model'

Arguments:
    <srcdir>: path to the data directory containing the original dataset
    <destdir>: path to the data directory for storing the enhanced dataset
    <modelfile>: path to the pretrained speech enhancement model
        Note: "train.yaml" is assumed to be in the same directory as <modelfile>

Optional:
    --audio_dir: specify the output directory for storing enhanced audios (default is '<destdir>/wavs')
    --scp_files: specify additional scp files to be copied and converted (default is '${scp_files}')
    --spk_num: number of speakers in the input (>1 for separation, default=${spk_num})
    --gpu_inference: whether to use gpu for inference (default=${gpu_inference})
    --inference_nj: The number of parallel jobs in inference (default=${inference_nj})
    --fs: sampling rate (default=${fs})
    --python: specify python to execute espnet commands (default=${python})
    --id_prefix: specify the prefix to prepend to utt ids and spk ids (default=${id_prefix})
    --enh_args: additional arguments for espnet2/bin/enh_inference.py (default is '${enh_args}')
EOF
)

log "$0 $*"

if [[ $# != 3 ]]; then
    log "${help_message}"
    exit 2
fi

srcdir=$1
destdir=$2
modelfile=$3


if [[ ! -f ${srcdir}/utt2spk ]]; then
    log "$0: no such file ${srcdir}/utt2spk"
    exit 1
fi

if [[ ${destdir} == "${srcdir}" ]]; then
    log "$0: this script requires <srcdir> and <destdir> to be different."
    exit 1
fi

if [[ ! -f "${modelfile}" ]]; then
    log "$0: no such file ${modelfile}"
    exit 1
fi
modeldir="$(dirname ${modelfile})"
if [[ ! -f "${modeldir}"/config.yaml ]]; then
    log "$0: no such file ${modeldir}/config.yaml"
    exit 1
fi

if [[ -z "${audio_dir}" ]]; then
  audio_dir="${destdir}/wavs"
fi

mkdir -p "${destdir}"
mkdir -p "${audio_dir}"

<"${srcdir}"/utt2spk awk -v p="${id_prefix}" '{printf("%s %s%s\n", $1, p, $1);}' > "${destdir}/utt_map"
<"${srcdir}"/spk2utt awk -v p="${id_prefix}" '{printf("%s %s%s\n", $1, p, $1);}' > "${destdir}/spk_map"
if [[ ! -f ${srcdir}/utt2uniq ]]; then
    <"${srcdir}/utt2spk" awk -v p="${id_prefix}" '{printf("%s%s %s\n", p, $1, $1);}' > "${destdir}/utt2uniq"
else
    <"${srcdir}/utt2uniq" awk -v p="${id_prefix}" '{printf("%s%s %s\n", p, $1, $2);}' > "${destdir}/utt2uniq"
fi


<"${srcdir}"/utt2spk utils/apply_map.pl -f 1 "${destdir}"/utt_map | \
  utils/apply_map.pl -f 2 "${destdir}"/spk_map >"${destdir}"/utt2spk

utils/utt2spk_to_spk2utt.pl <"${destdir}"/utt2spk >"${destdir}"/spk2utt

for scp_file in ${scp_files};do
    if [[ -f "${srcdir}/${scp_files}" ]]; then
        utils/apply_map.pl -f 1 "${destdir}"/utt_map <"${srcdir}/${scp_file}" >"${destdir}/${scp_file}"
    fi
done

for f in wav.scp text utt2lang; do
    if [[ -f ${srcdir}/${f} ]]; then
        utils/apply_map.pl -f 1 "${destdir}"/utt_map <"${srcdir}"/${f} >"${destdir}"/${f}
    fi
done
if [[ -f ${srcdir}/spk2gender ]]; then
    utils/apply_map.pl -f 1 "${destdir}"/spk_map <"${srcdir}"/spk2gender >"${destdir}"/spk2gender
fi

rm "${destdir}"/spk_map "${destdir}"/utt_map 2>/dev/null


# shellcheck disable=SC2154
if ${gpu_inference}; then
    _cmd=${cuda_cmd}
    _ngpu=1
else
    _cmd=${decode_cmd}
    _ngpu=0
fi

_logdir="$(realpath ${audio_dir})"
mkdir -p "${_logdir}"


# 1. Split the key file
key_file=${destdir}/wav.scp
split_scps=""
_nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
for n in $(seq "${_nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${key_file}" ${split_scps}

# 2. Submit inference jobs
log "Enhancement started... log: '${_logdir}/enhance_dataset.*.log'"
# shellcheck disable=SC2086
# TODO(wangyou): support enhancement from enh_asr models after https://github.com/espnet/espnet/pull/3226 is merged
${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/enhance_dataset.JOB.log \
    ${python} -m espnet2.bin.enh_inference \
        --ngpu "${_ngpu}" \
        --fs "${fs}" \
        --data_path_and_name_and_type "${destdir}/wav.scp,speech_mix,sound" \
        --key_file "${_logdir}"/keys.JOB.scp \
        --train_config "${modeldir}"/config.yaml \
        --model_file "${modelfile}" \
        --output_dir "${_logdir}"/output.JOB \
        ${enh_args}

_spk_list=" "
for i in $(seq ${spk_num}); do
    _spk_list+="spk${i} "
done

# 3. Concatenates the output files from each jobs
for spk in ${_spk_list}; do
    for i in $(seq "${_nj}"); do
        cat "${_logdir}/output.${i}/${spk}.scp"
    done | LC_ALL=C sort -k1 > "${_logdir}/${spk}.scp"
done

if [[ ${spk_num} -gt 1 ]]; then
    # (speech separation) prepare a subdir for each speaker
    for spk in ${_spk_list}; do
        mkdir -p "${destdir}/${spk}"
        cp "${_logdir}/${spk}.scp" "${destdir}/${spk}/wav.scp"
        for f in utt2spk spk2utt utt2lang; do
            if [[ -f "${destdir}/${f}" ]]; then
                ln -s ../${f} "${destdir}/${spk}/${f}"
            fi
        done
        for f in text spk2gender; do
            if [[ -f "${destdir}/${f}_${spk}" ]]; then
                ln -s ../${f}_${spk} "${destdir}/${spk}/${f}"
            fi
        done
        utils/validate_data_dir.sh --no-feats --no-text "${destdir}/${spk}"
    done
    log "$0: generated enhanced version of data in ${srcdir}, in ${destdir}/spk*"
else
    # (speech enhancement) no subdir is needed
    cp "${_logdir}/spk1.scp" "${destdir}/wav.scp"
    log "$0: generated enhanced version of data in ${srcdir}, in ${destdir}"
    utils/validate_data_dir.sh --no-feats --no-text "${destdir}"
fi
