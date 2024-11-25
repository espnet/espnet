#!/usr/bin/env bash

# Copyright 2024 Jiatong Shi
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
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts="" # Options to be passed to local/data.sh.

# Feature extraction related
feats_type=raw             # Input feature type.
audio_format=flac          # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
min_wav_duration=0.1       # Minimum duration in second.
max_wav_duration=30        # Maximum duration in second.
fs=16000                   # Sampling rate.

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
token_type=bpe      # Tokenization type (char or bpe) for text.
nbpe=30             # The number of BPE vocabulary for text.
bpemode=unigram     # Mode of BPE for text (unigram or bpe).
bpe_input_sentence_size=100000000 # Size of input sentence for BPE for text.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE of text.
bpe_char_cover=1.0  # character coverage when modeling BPE for text.

# Training related
train_config=""    # Config for training.
train_args=""      # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
universa_exp=""         # Specify the directory path for experiment. If this option is specified, tag is ignored.
universa_stats_dir=""   # Specify the directory path for statistics. If empty, automatically decided.
num_splits=1       # Number of splitting for universa corpus.
use_ref_wav=true   # Whether use reference wave or not.
use_ref_text=true  # Whether use reference text or not.
metric2id=    # File path for metric2id mapping.
metric2type=       # Metric type for metric2id mapping.

# Decoding related
inference_config="" # Config for decoding.
inference_args=""   # Arguments for decoding (e.g., "--threshold 0.75").
                    # Note that it will overwrite args in inference config.
inference_tag=""    # Suffix for decoding directory.
inference_model=valid.loss.best.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
download_model=""  # Download a model from Model Zoo and use it for decoding.

# Scoring related
sys_info=         # System information for system-level evaluation.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""     # Name of training set.
valid_set=""     # Name of validation set used for monitoring/tuning network training.
test_sets=""     # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
audio_fold_length=256000 # fold_length for audio data.

# Upload model related
hf_repo=

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>" --srctexts "<srctexts>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --fs               # Sampling rate (default="${fs}").

    # Tokenization related
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --token_type=bpe          # Tokenization type (char or bpe) for text. (default="${token_type}").
    --nbpe=30                 # The number of BPE vocabulary for text. (default="${nbpe}").
    --bpemode=unigram         # Mode of BPE for text (unigram or bpe). (default="${bpemode}").
    --bpe_input_sentence_size=100000000 # Size of input sentence for BPE for text. (default="${bpe_input_sentence_size}").
    --bpe_nlsyms=             # Non-linguistic symbols list, separated by a comma, for BPE of text. (default="${bpe_nlsyms}").
    --bpe_char_cover=1.0      # Character coverage when modeling BPE for text. (default="${bpe_char_cover}").


    # Training related
    --train_config  # Config for training (default="${train_config}").
    --train_args    # Arguments for training (default="${train_args}").
                    # e.g., --train_args "--max_epoch 1"
                    # Note that it will overwrite args in train config.
    --tag           # Suffix for training directory (default="${tag}").
    --universa_exp       # Specify the directory path for experiment.
                    # If this option is specified, tag is ignored (default="${universa_exp}").
    --universa_stats_dir # Specify the directory path for statistics.
                    # If empty, automatically decided (default="${universa_stats_dir}").
    --num_splits    # Number of splitting for universa corpus (default="${num_splits}").
    --use_ref_wav   # Whether use reference wave or not. (default="${use_ref_wav}").
    --use_ref_text  # Whether use reference text or not. (default="${use_ref_text}").
    --metric2id # File path for metric2id mapping (default="${metric2id}").
    --metric2type    # Metric type for metric2id mapping (default="${metric2type}").

    # Decoding related
    --inference_config  # Config for decoding (default="${inference_config}").
    --inference_args    # Arguments for decoding, (default="${inference_args}").
                        # e.g., --inference_args "--threshold 0.75"
                        # Note that it will overwrite args in inference config.
    --inference_tag     # Suffix for decoding directory (default="${inference_tag}").
    --inference_model   # Model path for decoding (default=${inference_model}).
    --download_model    # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # Scoring related
    --sys_info       # System information for system-level evaluation (default="${sys_info}").

    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set          # Name of training set (required).
    --valid_set          # Name of validation set used for monitoring/tuning network training (required).
    --test_sets          # Names of test sets (required).
                         # Note that multiple items (e.g., both dev and eval sets) can be specified.
    --bpe_train_text # Text file path of bpe training set.
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --audio_fold_length  # Fold length for audio data (default="${audio_fold_length}").

    # Upload model related
    ---hf_repo          # Huggingface model tag for huggingface model upload
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats="${dumpdir}/raw"
else
    log "${help_message}"
    log "Error: only supported: --feats_type raw"
    exit 2
fi

# Extra files for translation/synthesis process
utt_extra_files="metric.scp"
if [ ${use_ref_wav} = true ]; then
    utt_extra_files="${utt_extra_files} ref_wav.scp"
fi

[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/org/${train_set}/text"

# Check tokenization type
token_listdir=data/token_list
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)_${feats_type}_fs${fs}"
    else
        tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_$(echo "${inference_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for collect-stats mode
if [ -z "${universa_stats_dir}" ]; then
    universa_stats_dir="${expdir}/universa_stats_${feats_type}"
fi
# The directory used for training commands
if [ -z "${universa_exp}" ]; then
    universa_exp="${expdir}/universa_${tag}"
fi


# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi


    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # TODO(kamo): Change kaldi-ark to npy or HDF5?
        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi

            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

            for extra_file in ${utt_extra_files}; do
                cp "data/${dset}/${extra_file}" "${data_feats}${_suf}/${dset}"
            done

            _tgt_opts=
            if [ -e data/"${dset}"/segments ]; then
                _tgt_opts+="--segments data/${dset}/segments "
            fi
            log "Format target wav.scp"
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_tgt_opts} \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

            if [ ${use_ref_wav} = true ]; then
                _ref_opts=
                if [ -e data/"${dset}"/ref_segments ]; then
                    _ref_opts+="--segments data/${dset}/ref_segments "
                fi
                log "Format reference wav.scp"
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" --suffix ".ref" \
                    --out_filename "ref_wav.scp" ${_ref_opts} \
                    "data/${dset}/ref_wav.scp" "${data_feats}${_suf}/${dset}"
            fi

            # NOTE(jiatong): some extra treatment for extra files, including sorting and duplication remove
            for utt_extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py ${data_feats}${_suf}/${dset}/${utt_extra_file} \
                    > ${data_feats}${_suf}/${dset}/${utt_extra_file}.tmp
                mv ${data_feats}${_suf}/${dset}/${utt_extra_file}.tmp ${data_feats}${_suf}/${dset}/${utt_extra_file}
                sort -o ${data_feats}${_suf}/${dset}/${utt_extra_file} ${data_feats}${_suf}/${dset}/${utt_extra_file}
            done

            # TODO(jiatong): format preference wav.scp
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Prepare metric ID"

        if [ -z "${metric2id}" ] ; then
            if [ -z "${metric2type}" ] ; then
                log "metric2id and metric2type are not specificed. Generate it from metric.scp"
                python pyscripts/utils/prep_metric_id.py \
                    "${data_feats}/org/${train_set}"/metric.scp \
                    "${data_feats}/org/${train_set}"/metric2id
            else
                log "metric2id_file is not specificed but metric2type is. Generate it from metric2type"
                python pyscripts/utils/prep_metric_id.py \
                    "${data_feats}/org/${train_set}"/metric.scp \
                    "${data_feats}/org/${train_set}"/metric2id \
                    --metric_type ${metric2type}
            fi

        else
            log "metric2id is already specified. Skip this stage."
            cp -r "${data_feats}/org/${train_set}"/metric2id "${data_feats}/${train_set}"
        fi

    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            mkdir -p "${data_feats}/${dset}"
            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"

            if [ ${use_ref_wav} = true ]; then
                <"${data_feats}/org/${dset}/utt2num_samples.ref" \
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                        >"${data_feats}/${dset}/utt2num_samples.ref"
                <"${data_feats}/org/${dset}/ref_wav.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples.ref"  \
                    >"${data_feats}/${dset}/ref_wav.scp"
            fi

            # Remove empty text
            for utt_extra_file in ${utt_extra_files}; do
                <${data_feats}/org/${dset}/${utt_extra_file} \
                    awk ' { if( NF != 1 ) print $0; } ' > ${data_feats}/${dset}/${utt_extra_file}
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" "${data_feats}/${dset}"

            # NOTE(jiatong): some extra treatment for extra files, including sorting and duplication remove
            for utt_extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py ${data_feats}/${dset}/${utt_extra_file} \
                    > ${data_feats}/${dset}/${utt_extra_file}.tmp
                mv ${data_feats}/${dset}/${utt_extra_file}.tmp ${data_feats}/${dset}/${utt_extra_file}
		        sort -o ${data_feats}/${dset}/${utt_extra_file} ${data_feats}/${dset}/${utt_extra_file}
            done

        done

        # NOTE(jiatong): sync metric2id for train set
        cp "${data_feats}/org/${train_set}/metric2id" "${data_feats}/${train_set}/metric2id"
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && [ ${use_ref_text} = true ] ; then
        if [ "${token_type}" = bpe ]; then
            log "Stage 5: Generate token_list from ${bpe_train_text} using BPE"

            mkdir -p "${bpedir}"
            # shellcheck disable=SC2002
            cat ${bpe_train_text} | cut -f 2- -d" "  > "${bpedir}"/train.txt

            if [ -n "${bpe_nlsyms}" ]; then
                if test -f "${bpe_nlsyms}"; then
                    bpe_nlsyms_list=$(awk '{print $1}' ${bpe_nlsyms} | paste -s -d, -)
                    _opts_spm="--user_defined_symbols=${bpe_nlsyms_list}"
                else
                    _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
                fi
            else
                _opts_spm=""
            fi

            spm_train \
                --input="${bpedir}"/train.txt \
                --vocab_size="${nbpe}" \
                --model_type="${bpemode}" \
                --model_prefix="${bpeprefix}" \
                --character_coverage=${bpe_char_cover} \
                --input_sentence_size="${bpe_input_sentence_size}" \
                ${_opts_spm}

            {
            echo "${blank}"
            echo "${oov}"
            # Remove <unk>, <s>, </s> from the vocabulary
            <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
            echo "${sos_eos}"
            } > "${token_list}"

        elif [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
            log "Stage 5: Generate character level token_list from ${lm_train_text}"

            _opts="--non_linguistic_symbols ${nlsyms_txt}"

            # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
            # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
            ${python} -m espnet2.bin.tokenize_text  \
                --token_type "${token_type}" \
                --input "${data_feats}/lm_train.txt" --output "${token_list}" ${_opts} \
                --field 2- \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --write_vocabulary true \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"
        else
            log "Error: not supported --token_type '${token_type}'"
            exit 2
        fi

    fi

else
    log "Skip the stages for data preparation"
fi
# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: Universa collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To gen3 -m espnet2.bin.universa_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        # 1. Split the key file
        _logdir="${universa_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_train_dir}/${_scp} wc -l)" "$(<${_valid_dir}/${_scp} wc -l)")

        key_file="${_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${universa_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${universa_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${universa_stats_dir}/run.sh"; chmod +x "${universa_stats_dir}/run.sh"

        # 3. Submit jobs
        log "Universa collect_stats started... log: '${_logdir}/stats.*.log'"

        # Add reference audio and text if required
        if [ ${use_ref_wav} = true ]; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/ref_wav.scp,ref_audio,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/ref_wav.scp,ref_audio,${_type} "
        fi
        if [ ${use_ref_text} = true ]; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,ref_text,text "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,ref_text,text "
            _opts+="--token_list ${token_list} "
            _opts+="--token_type ${token_type} "
            _opts+="--bpemodel ${bpemodel} "
            _opts+="--cleaner ${cleaner} "
            _opts+="--non_linguistic_symbols ${nlsyms_txt} "
        fi
        if [ -n "${metric2type}" ]; then
            _opts+="--metric2type ${metric2type} "
        fi

        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m "espnet2.bin.universa_train" \
                --collect_stats true \
                --use_preprocessor true \
                --train_data_path_and_name_and_type "${_train_dir}/${_scp},audio,${_type}" \
                --train_data_path_and_name_and_type "${_train_dir}/metric.scp,metrics,metric" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${_scp},audio,${_type}" \
                --valid_data_path_and_name_and_type "${_valid_dir}/metric.scp,metrics,metric" \
                --metric2id "${data_feats}/${train_set}/metric2id" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${train_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        ${python} -m espnet2.bin.aggregate_stats_dirs --skip_sum_stats ${_opts} --output_dir "${universa_stats_dir}"
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 7: Universa Training: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.universa_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        log "Generate '${universa_exp}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${universa_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${universa_exp}/run.sh"; chmod +x "${universa_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "Universa training started... log: '${universa_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${universa_exp})"
        else
            jobname="${universa_exp}/train.log"
        fi

        # Add reference audio and text if required
        if [ ${use_ref_wav} = true ]; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/ref_wav.scp,ref_audio,${_type} "
            _opts+="--train_shape_file ${universa_stats_dir}/train/ref_audio_shape "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/ref_wav.scp,ref_audio,${_type} "
            _opts+="--valid_shape_file ${universa_stats_dir}/valid/ref_audio_shape "
        fi
        if [ ${use_ref_text} = true ]; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,ref_text,text "
            _opts+="--train_shape_file ${universa_stats_dir}/train/ref_text_shape "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,ref_text,text "
            _opts+="--valid_shape_file ${universa_stats_dir}/valid/ref_text_shape "
            _opts+="--token_list ${token_list} "
            _opts+="--token_type ${token_type} "
            _opts+="--bpemodel ${bpemodel} "
            _opts+="--cleaner ${cleaner} "
            _opts+="--non_linguistic_symbols ${nlsyms_txt} "
        fi
        if [ -n "${metric2type}" ]; then
            _opts+="--metric2type ${metric2type} "
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${universa_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${universa_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m "espnet2.bin.universa_train" \
                --use_preprocessor true \
                --resume true \
                --fold_length "${audio_fold_length}" \
                --train_data_path_and_name_and_type "${_train_dir}/${_scp},audio,${_type}" \
                --train_data_path_and_name_and_type "${_train_dir}/metric.scp,metrics,metric" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${_scp},audio,${_type}" \
                --valid_data_path_and_name_and_type "${_valid_dir}/metric.scp,metrics,metric" \
                --metric2id "${data_feats}/${train_set}/metric2id" \
                --train_shape_file ${universa_stats_dir}/train/audio_shape \
                --valid_shape_file ${universa_stats_dir}/valid/audio_shape \
                --output_dir "${universa_exp}" \
                ${_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    universa_exp="${expdir}/${download_model}"
    mkdir -p "${universa_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${universa_exp}/config.txt"

    # Get the path of each file
    _model_file=$(<"${universa_exp}/config.txt" sed -e "s/.*'model_file': '\([^']*\)'.*$/\1/")
    _train_config=$(<"${universa_exp}/config.txt" sed -e "s/.*'train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_model_file}" "${universa_exp}"
    ln -sf "${_train_config}" "${universa_exp}"
    inference_model=$(basename "${_model_file}")

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Decoding: training_dir=${universa_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        log "Generate '${universa_exp}/${inference_tag}/run.sh'. You can resume the process from stage 8 using this script"
        mkdir -p "${universa_exp}/${inference_tag}"; echo "${run_args} --stage 8 \"\$@\"; exit \$?" > "${universa_exp}/${inference_tag}/run.sh"; chmod +x "${universa_exp}/${inference_tag}/run.sh"


        for dset in ${test_sets}; do
            _opts=
            if [ -n "${inference_config}" ]; then
                _opts+="--config ${inference_config} "
            fi

            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi

            _data="${data_feats}/${dset}"
            _dir="${universa_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/log"
            mkdir -p "${_logdir}"

            # 0. Copy feats_type
            cp "${_data}/feats_type" "${_dir}/feats_type"

            # 1. Split the key file
            key_file=${_data}/wav.scp
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/universa_inference.*.log'"

            # Add reference audio and text if required
            if [ ${use_ref_wav} = true ]; then
                _opts+="--data_path_and_name_and_type ${_data}/ref_wav.scp,ref_audio,${_type} "
            fi
            if [ ${use_ref_text} = true ]; then
                _opts+="--data_path_and_name_and_type ${_data}/text,ref_text,text "
            fi

            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/universa_inference.JOB.log \
                ${python} -m espnet2.bin.universa_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type ${_data}/${_scp},audio,${_type} \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --model_file "${universa_exp}"/"${inference_model}" \
                    --train_config "${universa_exp}"/config.yaml \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/universa_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            # shellcheck disable=SC2068
            for n in $(seq "${_nj}"); do
                cat "${_logdir}"/output.${n}/metric.scp || exit 1
            done > "${_dir}/metric.scp"
        done
    fi

    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Scoring"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _ref_metrics="${_data}/metric.scp"
            _dir="${universa_exp}/${inference_tag}/${dset}"
            _pred_metrics="${_dir}/metric.scp"

            log "Begin evaluation on ${dset}, results are written under ${_dir}"

            log "Perform utt-level evaluation, results are written in ${_dir}/utt_result.json"
            python pyscripts/utils/universa_eval.py \
                --level utt \
                --ref_metrics "${_ref_metrics}" \
                --pred_metrics "${_pred_metrics}" \
                --skip_missing true \
                --out_file "${_dir}/utt_result.json"
            
            log "Utterance-level evaluation results are as follows:"
            cat "${_dir}/utt_result.json"

            if [ -n "${sys_info}" ]; then
                log "Perform system-level evaluation using ${sys_info}, results are written in ${_dir}/sys_result.json"
                python pyscripts/utils/universa_eval.py \
                    --level sys \
                    --ref_metrics "${_ref_metrics}" \
                    --pred_metrics "${_pred_metrics}" \
                    --sys_info "${sys_info}" \
                    --skip_missing true \
                    --out_file "${_dir}/sys_result.json"
                
                log "System-level evaluation results are as follows:"
                cat "${_dir}/sys_result.json"
            fi

        done
    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${universa_exp}/${universa_exp##*/}_${inference_model%.*}.zip"
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! "${skip_upload}"; then
    log "Stage 10: Packing model: ${packed_model}"
    # Pack model
    if [ -e "${universa_exp}/${inference_model}" ]; then
        # Pack model
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack universa \
            --model_file "${universa_exp}/${inference_model}" \
            --train_config "${universa_exp}/config.yaml" \
            --option "${universa_exp}/images" \
            --outpath "${packed_model}"
    else
        log "Skip packing model since ${universa_exp}/${inference_model} does not exist."
    fi
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ] && ! "${skip_upload}"; then
    log "Stage 11: Uploading to hugging face: ${hf_repo}"
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1

    # Upload model to hugging face
    if [ -e "${packed_model}" ]; then

    gitlfs=$(git lfs --version 2> /dev/null || true)
    [ -z "${gitlfs}" ] && \
        log "ERROR: You need to install git-lfs first" && \
        exit 1

    dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
    [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="git checkout $(git show -s --format=%H)"
    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/universa1/ -> foo/universa1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/asr1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=universa
    # shellcheck disable=SC2034
    espnet_task=universa
    # shellcheck disable=SC2034
    lang=multilingual
    # shellcheck disable=SC2034
    task_exp=${universa_exp}
    eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

    this_folder=${PWD}
    cd ${dir_repo}
    if [ -n "$(git status --porcelain)" ]; then
        git add .
        git commit -m "Update model"
    fi
    git push
    cd ${this_folder}
    else
        log "Skip uploading to hugging face since ${packed_model} does not exist. Please run Stage 10 first."
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
