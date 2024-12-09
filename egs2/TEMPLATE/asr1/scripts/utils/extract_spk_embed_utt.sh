#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

nj=32
cmd="run.pl"
gpu=0 # TODO(jhan): verify this on GPU, currently only CPU is tested
pretrained_model=espnet/voxcelebs12_rawnet3
toolkit=espnet
spk_embed_tag=espnet_spk
data=
output=

log "$0 $@" # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1

# check arguments
if [ -z "${data}" ]; then
    log "Error: --data is not specified"
    exit 1
fi
if [ -z "${output}" ]; then
    log "Error: --output is not specified"
    exit 1
fi

logdir="${output}/logdir"
wav_scp="${data}/wav.scp"

rm -r "${logdir}"/data.* "${logdir}"/spk_embed_extract.* 2>/dev/null || true


_scp="${wav_scp}"
# create split
split_scps=""
for n in $(seq "${nj}"); do
    mkdir -p "${logdir}/data.${n}"
    split_scps+=" ${logdir}/data.${n}/$(basename "${_scp}")"
done

perl utils/split_scp.pl ${_scp} ${split_scps}


${cmd} --gpu "${gpu}" JOB=1:${nj} \
    "${logdir}/spk_embed_extract.JOB.log" \
    pyscripts/utils/extract_spk_embed_utt.py \
    --pretrained_model "${pretrained_model}" \
    --toolkit "${toolkit}" \
    --spk_embed_tag "${spk_embed_tag}" \
    "${logdir}/data.JOB" \
    "${logdir}/spk_embed_extract.JOB"

# combine scp files
for n in $(seq $nj); do
    cat "${logdir}/spk_embed_extract.${n}/${spk_embed_tag}.scp" || exit 1
done >"${output}/${spk_embed_tag}.scp" || exit 1

# # combine ark files
# ark_files=""
# for n in $(seq $nj); do
#     ark_files+=" ${logdir}/spk_embed_extract.${n}/${spk_embed_tag}.ark"
# done
# copy-vector ark:<(cat $ark_files) ark:"${output}/${spk_embed_tag}.ark" || exit 1
