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

# Core settings
datadir=
outdir=

# General configuration
stage=1
stop_stage=3
nj=8
gpu_inference=false
fs=16000
do_data_prep=false
scp_suffix=
tgt_lang=
cachedir=".cache"
remove_formated_data=false

# Model related configuration
model_tag=""
asr_model_file=""
lm_file=""

# Inference option related configuration
inference_config=""
inference_args=""

# Scoring related configuration
nlsyms_txt=none
cleaner=none
gt_text=""

help_message=$(cat << EOF
Usage: $0 [Options]

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --nj             # Number of parallel jobs (default="${nj}").
    --gpu_inference  # Whether to use gpu in the inference (default="${gpu_inference}").
    --fs             # Sampling rate for ASR model inputs (default="${fs}").

    # Model related configuration
    --model_tag       # Model tag or url available in espnet_model_zoo (default="${model_tag}")
                      # If provided, overwrite --asr_model_file and --lm_file options.
    --asr_model_file  # ASR model file path in local (default="${asr_model_file}").
    --lm_file         # LM model file path in local (default="${lm_file}").

    # Inference related configuration
    --inference_config  # ASR inference configuration file (default="${inference_config}").
    --inference_args    # Additional arguments for ASR inference (default=${inference_args}).

    # Scoring related configuration
    --nlsyms_txt  # Non-language symbol file (default="${nlsyms_txt}").
    --cleaner     # Text cleaner module for the reference (default="${cleaner}").
    --gt_text     # Kaldi-format groundtruth text file (default="${gt_text}")
                  # This must be provided if you want to calculate scores.

Examples:
    # Use pretrained model and perform only inference
    $0 --model_tag <model_tag> kaldi_datadir asr_outputs --stop-stage 2

    # Use pretrained model and perform inference and scoring
    $0 --model_tag <model_tag> --stop-stage 3 --gt_text /path/to/text kaldi_datadir asr_results

    # Use local model and perform inference and scoring
    $0 --asr_model_file /path/to/model.pth --stop-stage 3 --gt_text /path/to/text kaldi_datadir asr_results

EOF
)

log "$0 $*"
# shellcheck disable=SC1091
. utils/parse_options.sh

if [ -z "${datadir}" ] || [ -z "${outdir}" ] ; then
    log "${help_message}"
    exit 2
fi

# shellcheck disable=SC1091
. ./path.sh
# shellcheck disable=SC1091
. ./cmd.sh

# Check the option is valid
if [ -z "${gt_text}" ] && [ ! -e "${datadir}/text${scp_suffix}" ]; then
    log "--gt_text must be provided if perform scoring."
    exit 1
fi
if [ -z "${model_tag}" ] && [ -z "${asr_model_file}" ]; then
    log "Either --model_tag or --asr_model_file must be provided."
    exit 1
fi

if ${gpu_inference}; then
    # shellcheck disable=SC2154
    _cmd="${cuda_cmd}"
    _ngpu=1
else
    # shellcheck disable=SC2154
    _cmd="${decode_cmd}"
    _ngpu=0
fi

# NOTE(jiatong): we assume the formated data,
#                but we also provide options to prepare data from scratch
source_wavscp="${outdir}/wav/wav.scp"

if ${do_data_prep}; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "stage 1: Format wav.scp"
        # shellcheck disable=SC2154
        scripts/audio/format_wav_scp.sh \
            --nj "${nj}" \
            --cmd "${train_cmd}" \
            --audio-format wav \
            --fs "${fs}" \
            "${outdir}/wav/wav.scp" "${outdir}/tmp"
    fi
    source_wavscp="${outdir}/tmp/wav.scp"
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
        espnet_model_zoo_download --unpack true "${model_tag}" --cachedir "${cachedir}"> /dev/null
        _opts+=("--model_tag" "${model_tag}")
    fi

    logdir="${outdir}/logdir"
    mkdir -p "${logdir}"

    # 1. Split the key file
    key_file=${source_wavscp}
    split_scps=""
    _nj=$(min "${nj}" "$(wc -l < "${key_file}")")
    for n in $(seq "${_nj}"); do
        split_scps+=" ${logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Submit decoding jobs
    log "Decoding started... log: '${logdir}/asr_inference.*.log'"
    # shellcheck disable=SC2046,SC2086
    ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${logdir}"/asr_inference.JOB.log \
        python3 -m espnet2.bin.asr_inference \
            --ngpu "${_ngpu}" \
            --data_path_and_name_and_type "${source_wavscp},speech,sound" \
            --key_file "${logdir}"/keys.JOB.scp \
            --output_dir "${logdir}"/output.JOB \
            "${_opts[@]}" ${inference_args} || { cat $(grep -l -i error "${logdir}"/asr_inference.*.log) ; exit 1; }

    # 3. Concatenates the output files from each jobs
    for f in token token_int score text; do
        for i in $(seq "${_nj}"); do
            cat "${logdir}/output.${i}/1best_recog/${f}"
        done | LC_ALL=C sort -k1 >"${outdir}/${f}"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Scoring"

    _scoredir="${outdir}/score_asr_bleu"
    mkdir -p "${_scoredir}"

    if [ -z ${gt_text} ]; then
        ref_text=${datadir}/text${scp_suffix}
    else
        ref_text=${gt_text}
    fi

    log "${ref_text}"
    log "${datadir}"

    paste \
        <(<"${ref_text}" \
            python3 -m espnet2.bin.tokenize_text  \
                -f 2- --input - --output - \
                --token_type word \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --remove_non_linguistic_symbols true \
                --cleaner "${cleaner}" \
                ) \
        <(<"${datadir}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >"${_scoredir}/ref.trn.org"

    paste \
        <(<"${outdir}/text"  \
                python3 -m espnet2.bin.tokenize_text  \
                    -f 2- --input - --output - \
                    --token_type word \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --remove_non_linguistic_symbols true \
                    ) \
        <(<"${source_wavscp}" awk '{ print "(" $1 ")" }') \
            >"${_scoredir}/hyp.trn.org"

    # remove utterance id
    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/ref.trn.org" > "${_scoredir}/ref.trn"
    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/hyp.trn.org" > "${_scoredir}/hyp.trn"

    if [ -n ${tgt_lang} ]; then
        # detokenizer
        detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/ref.trn" > "${_scoredir}/ref.trn.detok"
        detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/hyp.trn" > "${_scoredir}/hyp.trn.detok"
    else
        cp "${_scoredir}/ref.trn" "${_scoredir}/ref.trn.detok"
        cp "${_scoredir}/hyp.trn" "${_scoredir}/hyp.trn.detok"
    fi

    # detokenize & remove punctuation except apostrophe
    remove_punctuation.pl < "${_scoredir}/ref.trn.detok" > "${_scoredir}/ref.trn.detok.lc.rm"
    remove_punctuation.pl < "${_scoredir}/hyp.trn.detok" > "${_scoredir}/hyp.trn.detok.lc.rm"

    echo "Case insensitive BLEU result (single-reference)" > ${_scoredir}/result.lc.txt
    sacrebleu -lc "${_scoredir}/ref.trn.detok.lc.rm" \
                -i "${_scoredir}/hyp.trn.detok.lc.rm" \
                -m bleu chrf ter \
                >> ${_scoredir}/result.lc.txt
    log "Write a case-insensitve BLEU (single-reference) result in ${_scoredir}/result.lc.txt"

    # TODO(jiatong): add multi-references cases

fi

# Remove tmp dir if exists
if "${remove_formated_data}"; then
    rm -rf "${outdir}/tmp"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
