#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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
SECONDS=0

# General configuration
stage=1
stop_stage=2
nj=8
inference_nj=8
gpu_inference=false
fs=16000

# Model related configuration
model_tag=""
asr_model_file=""
lm_file=""
whisper_tag=""
whisper_dir=""

# Inference option related configuration
inference_config=""
inference_args=""
## change the language id according to your dataset
decode_options="{task: transcribe, language: en, beam_size: 1}"

# Scoring related configuration
bpemodel=""
nlsyms_txt=none
cleaner=none
hyp_cleaner=none
gt_text=""

help_message=$(cat << EOF
Usage: $0 [Options] <wav.scp> <outdir>

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --nj             # Number of parallel jobs (default="${nj}").
    --inference_nj   # Number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference  # Whether to use gpu in the inference (default="${gpu_inference}").
    --fs             # Sampling rate for ASR model inputs (default="${fs}").

    # Model related configuration
    --model_tag       # Model tag or url available in espnet_model_zoo (default="${model_tag}")
                      # If provided, overwrite --asr_model_file and --lm_file options.
    --asr_model_file  # ASR model file path in local (default="${asr_model_file}").
    --lm_file         # LM model file path in local (default="${lm_file}").
    --whisper_tag     # Whisper model tag for evaluation with Whisper (default="${whisper_tag}").
    --whisper_dir     # Whisper model directory to download (default="${whisper_dir}").

    # Inference related configuration
    --inference_config  # ASR inference configuration file (default="${inference_config}").
    --inference_args    # Additional arguments for ASR inference (default=${inference_args}).
    --decode_options    # Decode options for Whisper's transcribe method (default=${decode_options}).

    # Scoring related configuration
    --bpemodel    # BPE model path, needed if you want to calculate TER (default="${bpemodel}").
    --nlsyms_txt  # Non-language symbol file (default="${nlsyms_txt}").
    --cleaner     # Text cleaner module for the reference (default="${cleaner}").
    --hyp_cleaner # Text cleaner module for the hypothesis (default="${hyp_cleaner}").
    --gt_text     # Kaldi-format groundtruth text file (default="${gt_text}")
                  # This must be provided if you want to calculate scores.

Examples:
    # Use pretrained model and perform only inference
    $0 --model_tag <model_tag> wav.scp asr_outputs

    # Use pretrained model and perform inference and scoring
    $0 --model_tag <model_tag> --stop-stage 3 --gt_text /path/to/text wav.scp asr_results

    # Use local model and perform inference and scoring
    $0 --asr_model_file /path/to/model.pth --stop-stage 3 --gt_text /path/to/text wav.scp asr_results

    # Use whisper model and perform inference and scoring
    $0 --whisper_tag small --whisper_dir /path/to/download --decode_options "{task: transcribe; language: en}" \
        --stop-stage 3 --gt_text /path/to/text wav.scp asr_results

EOF
)

log "$0 $*"
# shellcheck disable=SC1091
. utils/parse_options.sh

wavscp=$1
outdir=$2

if [ $# -ne 2 ]; then
    log "${help_message}"
    exit 2
fi

# shellcheck disable=SC1091
. ./path.sh
# shellcheck disable=SC1091
. ./cmd.sh

# Check the option is valid
if [ -z "${gt_text}" ] && [ "${stop_stage}" -ge 3 ]; then
    log "--gt_text must be provided if perform scoring."
    exit 1
fi
if [ -z "${model_tag}" ] && [ -z "${asr_model_file}" ] && [ -z "${whisper_tag}" ]; then
    log "--model_tag or --asr_model_file or --whisper_tag must be provided."
    exit 1
fi

if ${gpu_inference}; then
    # shellcheck disable=SC2154
    _cmd="${cuda_cmd}"
    _ngpu=1
    inference_nj=1
else
    # shellcheck disable=SC2154
    _cmd="${decode_cmd}"
    _ngpu=0
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Format wav.scp"
    # shellcheck disable=SC2154
    scripts/audio/format_wav_scp.sh \
        --nj "${nj}" \
        --cmd "${train_cmd}" \
        --audio-format wav \
        --fs "${fs}" \
        "${wavscp}" "${outdir}/tmp"
fi

if [ -e "${outdir}/tmp/wav.scp" ]; then
    wavscp="${outdir}/tmp/wav.scp"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: ASR inference"
    _opts=()
    if [ -n "${inference_config}" ]; then
        _opts+=("--config" "${inference_config}")
    fi
    if [ -n "${asr_model_file}" ]; then
        _opts+=("--asr_model_file" "${asr_model_file}")
    fi
    if [ -n "${lm_file}" ]; then
        _opts+=("--lm_file" "${lm_file}")
    fi
    if [ -n "${model_tag}" ]; then
        # FIXME: workaround until fixing filelock in espnet_model_zoo
        espnet_model_zoo_download --unpack true "${model_tag}" > /dev/null
        _opts+=("--model_tag" "${model_tag}")
    fi

    logdir="${outdir}/logdir"
    mkdir -p "${logdir}"

    # 1. Split the key file
    key_file=${wavscp}
    split_scps=""
    _nj=$(min "${inference_nj}" "$(wc -l < "${key_file}")")
    for n in $(seq "${_nj}"); do
        split_scps+=" ${logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Submit decoding jobs
    log "Decoding started... log: '${logdir}/asr_inference.*.log'"

    if [ -n "${whisper_tag}" ]; then
        if [ -z "${whisper_dir}" ]; then
            whisper_dir=${outdir}/models
        fi
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${logdir}"/asr_inference.JOB.log \
            python3 pyscripts/utils/evaluate_whisper_inference.py \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${wavscp}" \
                --key_file "${logdir}"/keys.JOB.scp \
                --model_tag ${whisper_tag} \
                --model_dir ${whisper_dir} \
                --output_dir "${logdir}"/output.JOB \
                --decode_options "${decode_options}" || { cat $(grep -l -i error "${logdir}"/asr_inference.*.log) ; exit 1; }
    else
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${logdir}"/asr_inference.JOB.log \
            python3 -m espnet2.bin.asr_inference \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${wavscp},speech,sound" \
                --key_file "${logdir}"/keys.JOB.scp \
                --output_dir "${logdir}"/output.JOB \
                "${_opts[@]}" ${inference_args} || { cat $(grep -l -i error "${logdir}"/asr_inference.*.log) ; exit 1; }
    fi

    # 3. Concatenates the output files from each jobs
    for f in token token_int score text; do
        if [ -f "${logdir}/output.1/1best_recog/${f}" ]; then
            for i in $(seq "${_nj}"); do
                cat "${logdir}/output.${i}/1best_recog/${f}"
            done | LC_ALL=C sort -k1 >"${outdir}/${f}"
        fi
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Scoring"
    for _type in cer wer ter; do
        [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

        _scoredir="${outdir}/score_${_type}"
        mkdir -p "${_scoredir}"

        if [ "${_type}" = wer ]; then
            paste \
                <(<"${gt_text}" \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type word \
                          --non_linguistic_symbols "${nlsyms_txt}" \
                          --remove_non_linguistic_symbols true \
                          --cleaner "${cleaner}" \
                          ) \
                <(<"${wavscp}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/ref.trn"
            paste \
                <(<"${outdir}/text"  \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type word \
                          --non_linguistic_symbols "${nlsyms_txt}" \
                          --remove_non_linguistic_symbols true \
                          --cleaner "${hyp_cleaner}" \
                          ) \
                <(<"${wavscp}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/hyp.trn"

        elif [ "${_type}" = cer ]; then
            paste \
                <(<"${gt_text}" \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type char \
                          --non_linguistic_symbols "${nlsyms_txt}" \
                          --remove_non_linguistic_symbols true \
                          --cleaner "${cleaner}" \
                          ) \
                <(<"${wavscp}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/ref.trn"
            paste \
                <(<"${outdir}/text" \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type char \
                          --non_linguistic_symbols "${nlsyms_txt}" \
                          --remove_non_linguistic_symbols true \
                          --cleaner "${hyp_cleaner}" \
                          ) \
                <(<"${wavscp}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/hyp.trn"

        elif [ "${_type}" = ter ]; then
            paste \
                <(<"${gt_text}" \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type bpe \
                          --bpemodel "${bpemodel}" \
                          --cleaner "${cleaner}" \
                        ) \
                <(<"${wavscp}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/ref.trn"
            paste \
                <(<"${outdir}/text" \
                      python3 -m espnet2.bin.tokenize_text  \
                          -f 2- --input - --output - \
                          --token_type bpe \
                          --bpemodel "${bpemodel}" \
                          --cleaner "${hyp_cleaner}" \
                          ) \
                <(<"${wavscp}" awk '{ print "(" $1 ")" }') \
                    >"${_scoredir}/hyp.trn"

        fi

        # Scoring
        sclite \
            -r "${_scoredir}/ref.trn" trn \
            -h "${_scoredir}/hyp.trn" trn \
            -i rm -o all stdout > "${_scoredir}/result.txt"

        log "Write ${_type} result in ${_scoredir}/result.txt"
        grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
    done
fi

# Remove tmp dir if exists
rm -rf "${outdir}/tmp"

log "Successfully finished. [elapsed=${SECONDS}s]"
