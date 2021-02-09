#!/bin/bash
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

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.

# Speed perturbation related
speed_perturb_factors="0.9 1.0 1.1"  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=5000             # The number of BPE vocabulary.

# ASR model related
asr_exp=
asr_tag=
asr_config=conf/tuning/train_asr_conformer.yaml

# Decoding related
inference_tag=    # Suffix to the result dir for decoding.
inference_config=conf/decode_asr.yaml

# [Task dependent] Set the datadir name created by local/data.sh
#test_sets="test_clean test_other dev_clean dev_other"
test_sets="test_clean"
lang=en

# Huggingface pretrained model card example
huggingface_example="byan/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp"

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

if [ -z "${asr_tag}" ]; then
    if [ -n "${asr_config}" ]; then
        asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
    else
        asr_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        asr_tag+="_${lang}_${token_type}"
    else
        asr_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_tag+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_tag+="_sp"
    fi
fi
data_feats=${dumpdir}/raw

# The directory used for training commands
if [ -z "${asr_exp}" ]; then                                                                                                                                                              
    asr_exp="${expdir}/asr_${asr_tag}"
fi

# Data download / prep
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for test sets"
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh --test_only true --stop_stage 2
fi

# Feature generation
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

    # ====== Recreating "wav.scp" ======
    # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
    # shouldn't be used in training process.
    # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
    # and it can also change the audio-format and sampling rate.
    # If nothing is need, then format_wav_scp.sh does nothing:
    # i.e. the input file format and rate is same as the output.

    for dset in ${test_sets}; do
        _suf=""
        utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
        rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
        _opts=
        if [ -e data/"${dset}"/segments ]; then
            # "segments" is used for splitting wav files which are written in "wav".scp
            # into utterances. The file format of segments:
            #   <segment_id> <record_id> <start_time> <end_time>
            #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
            # Where the time is written in seconds.
            _opts+="--segments data/${dset}/segments "
        fi
        # shellcheck disable=SC2086     
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
            "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

        echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
    done
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    log "Stage 11: Decoding: training_dir=${asr_exp}"

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _opts=
    if [ -n "${inference_config}" ]; then
        _opts+="--config ${inference_config} "
    fi

    for dset in ${test_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${asr_exp}/${inference_tag}/${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        _feats_type="$(<${_data}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
        else
            _scp=feats.scp
            _type=kaldi_ark
        fi

        # 1. Split the key file
        key_file=${_data}/${_scp}
        split_scps=""
        _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
        # shellcheck disable=SC2086
        # --asr_model_file could be a dummy string if and only if --pretrained_huggingface_id is specified
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            ${python} -m espnet2.bin.asr_inference \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --asr_train_config ${asr_config} \
                --asr_model_file "dummy_value" \
                --pretrained_huggingface_id ${huggingface_example} \
                --output_dir "${_logdir}"/output.JOB

        #${python} -m espnet2.bin.asr_inference \
        #    --ngpu "${_ngpu}" \
        #    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
        #    --key_file "${_logdir}"/keys.1.scp \
        #    --asr_train_config ${asr_config} \
        #    --asr_model_file "dummy_value" \
        #    --pretrained_huggingface_id ${huggingface_example} \
        #    --output_dir "${_logdir}"/output.1
        
        # 3. Concatenates the output files from each jobs
        for f in token token_int score text; do
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/1best_recog/${f}"
            done | LC_ALL=C sort -k1 >"${_dir}/${f}"
        done
    done
fi
