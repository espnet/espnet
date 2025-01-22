#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

nj=4                # number of parallel jobs
python=python3      # Specify python to execute espnet commands.
codec_choice=beats  # Audio Tokenizer Options: beats
codec_fs=16000
code_writeformat=text # ark or text
batch_size=1        # BEATs Audio tokenizaiton supports only batch_size=1
bias=0
dump_audio=false
file_name=
src_dir=
tgt_dir=
checkpoint_path=
config_path=
cuda_cmd=utils/run.pl

log "$0 $*"
. utils/parse_options.sh

. ./path.sh || exit 1
. ./cmd.sh || exit 1

if [ $# -ne 0 ]; then
    echo "Usage: $0 --src_dir <src_dir> --tgt_dir <tgt_dir> --file_name wav.scp --codec_choice beats"
    exit 0
fi

# If src_dir and tgt_dir are same then it will overwrite
# the files in src_dir so give warning
if [ ${src_dir} == ${tgt_dir} ]; then
    log "src_dir and tgt_dir are same. This script will overwrite the files in src_dir."
fi

if [[ ${file_name} == *.scp ]]; then
    file_name="${file_name%.scp}"
else
    echo "file_name should end with .scp suffix. ${file_name}"
fi

output_dir=${tgt_dir}/data
mkdir -p "${output_dir}"
_logdir=${tgt_dir}/logdir
mkdir -p "${_logdir}"
mkdir -p ${tgt_dir}/token_lists/

nutt=$(<"${src_dir}"/${file_name}.scp wc -l)
_nj=$((nj<nutt?nj:nutt))

split_scps=""
for n in $(seq ${_nj}); do
    split_scps+=" ${_logdir}/${file_name}.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl ${src_dir}/${file_name}.scp ${split_scps} || exit 1;
_opts=""
if [ ${config_path} ]; then
    _opts+="--config_path ${config_path} "
fi
if [ ${checkpoint_path} ]; then
    _opts+="--checkpoint_path ${checkpoint_path} "
fi

wav_wspecifier="ark,scp:${output_dir}/${file_name}_resyn_${codec_choice}.JOB.ark,${output_dir}/${file_name}_resyn_${codec_choice}.JOB.scp"
if [ ${code_writeformat} == "ark" ]; then
    code_wspecifier="ark,scp:${output_dir}/${file_name}_codec_${codec_choice}.JOB.ark,${output_dir}/${file_name}_codec_${codec_choice}.JOB.scp"
    _combine_filetype=scp
elif [ ${code_writeformat} == "text" ]; then
    code_wspecifier="ark,t:${output_dir}/${file_name}_codec_${codec_choice}.JOB.txt"
    _combine_filetype=txt
else
    echo "Error: Unsupported code_writeformat=${code_writeformat}"
    exit 1
fi

${cuda_cmd} --gpu 1 JOB=1:${_nj} ${_logdir}/codec_dump_${codec_choice}.JOB.log \
    ${python} pyscripts/feats/dump_codec.py \
        --codec_choice ${codec_choice} \
        --codec_fs ${codec_fs} \
        --batch_size ${batch_size} \
        --bias ${bias} \
        --dump_audio ${dump_audio} \
        --rank JOB \
        --vocab_file ${tgt_dir}/token_lists/codec_token_list \
        --wav_wspecifier ${wav_wspecifier} \
        ${_opts} \
        "scp:${_logdir}/${file_name}.JOB.scp" ${code_wspecifier} || exit 1;

for n in $(seq ${_nj}); do
    cat ${output_dir}/${file_name}_codec_${codec_choice}.${n}.${_combine_filetype} || exit 1;
done > ${tgt_dir}/${file_name}_${codec_choice}.${_combine_filetype} || exit 1

if [ ${code_writeformat} == "text" ]; then
    # format text file: [ 2 4 5 6 ] -> 2 4 5 6
    sed -i 's/[][]//g' ${tgt_dir}/${file_name}_${codec_choice}.${_combine_filetype}
    # remove multiple spaces
    sed -i 's/  */ /g' ${tgt_dir}/${file_name}_${codec_choice}.${_combine_filetype}
fi

if ${dump_audio}; then
    for n in $(seq ${_nj}); do
        cat ${output_dir}/${file_name}_resyn_${codec_choice}.${n}.scp || exit 1;
    done > ${tgt_dir}/${file_name}_resyn_${codec_choice}.scp || exit 1
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
