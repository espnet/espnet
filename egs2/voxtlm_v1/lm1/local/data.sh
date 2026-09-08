#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000

# Only affect the *train* split of speechlm/textlm (dev/test stay full-size,
# same as the paper: only train-set volume changes across Bal/3M/Set).
# Empty (default) = keep full data, i.e. the DSet config.
speechlm_train_size=
textlm_train_size=

# Convenience preset matching Table 2 of the VoxtLM paper (arXiv:2309.07937):
#   bal: speechlm=300000 textlm=300000   (D_Bal)
#   3m:  speechlm=3000000 textlm=3000000 (D_3M)
#   set: full data, no subsampling       (D_Set; same as leaving this unset)
# Explicit --speechlm_train_size/--textlm_train_size always win over this preset.
data_config=

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -n "${data_config}" ]; then
    case "${data_config}" in
        bal)
            [ -z "${speechlm_train_size}" ] && speechlm_train_size=300000
            [ -z "${textlm_train_size}" ] && textlm_train_size=300000
            ;;
        3m)
            [ -z "${speechlm_train_size}" ] && speechlm_train_size=3000000
            [ -z "${textlm_train_size}" ] && textlm_train_size=3000000
            ;;
        set)
            : # full data, nothing to do
            ;;
        *)
            log "Error: unknown --data_config '${data_config}' (expected bal, 3m, or set)"
            exit 2
            ;;
    esac
    log "data_config=${data_config}: speechlm_train_size=${speechlm_train_size:-full} textlm_train_size=${textlm_train_size:-full}"
fi

#data_dict={"LIBRISPEECH":"asr", "LIBRITTS":"tts", "VCTK":"tts", "LIBRISPEECH":"lm", "LIBRILIGHT":"lm"}

declare -a arr=("librispeech" "libritts" "vctk" "librilight")
declare -a arr_task=("asr" "tts" "textlm" "speechlm")

data_root="data" #/temp"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for dataset in "${arr[@]}"
    do
        echo $dataset
        if ! [ -f local/data_${dataset}.sh ]; then
            echo "File local/data_${dataset}.sh does not exist."
            exit 1
        else
            ./local/data_${dataset}.sh "${data_root}/${dataset}"
        fi
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"

    # Combine all tasks into "train/ dev/ test"
    for _dset in "train" "dev" "test"; do
        for task in "${arr_task[@]}"; do
            speech_dirs=""
            text_data_paths=""
            for dataset in "${arr[@]}"; do
                # check if dataset-task pair exists
                src_path=${data_root}/${dataset}/${task}/${_dset}
                # echo "Checking $src_path"
                if [ -d "$src_path" ]; then
                    # text
                    if [ -f "${src_path}/text" ] &&  [ ! -f "${src_path}/wav.scp" ]; then
                        text_data_paths+=" ${src_path}/text"
                    fi

                    # speech
                    if [ -f "${src_path}/wav.scp" ]; then
                        speech_dirs+=" ${src_path}"
                        if [ -f "${src_path}/text" ]; then
                            utils/validate_data_dir.sh --no-feats ${src_path}
                        else
                            utils/validate_data_dir.sh --no-feats --no-text ${src_path}
                        fi
                    fi
                fi
            done

            # combine speech dirs
            if [ -n "${speech_dirs}" ]; then
                if [ "${task}" = "speechlm" ] && [ "${_dset}" = "train" ] && [ -n "${speechlm_train_size}" ]; then
                    _combine_dir="${data_root}/.combine_${task}_${_dset}"
                    rm -rf "${_combine_dir}"
                    utils/combine_data.sh "${_combine_dir}" ${speech_dirs}
                    _nutt=$(<"${_combine_dir}"/wav.scp wc -l)
                    if [ "${_nutt}" -le "${speechlm_train_size}" ]; then
                        log "speechlm train has ${_nutt} utts, <= requested ${speechlm_train_size}. Keeping all."
                        rm -rf "data/${_dset}/speech/${task}"
                        mv "${_combine_dir}" "data/${_dset}/speech/${task}"
                    else
                        log "Subsampling speechlm train: ${_nutt} -> ${speechlm_train_size} utts"
                        mkdir -p "${_combine_dir}/logdir"
                        # Write shuf's full output to a file first (rather than piping
                        # straight into head), so head exiting early once it has enough
                        # lines doesn't SIGPIPE the still-writing shuf process and fail
                        # the whole pipeline under `set -o pipefail`.
                        shuf --random-source=<(yes "speechlm_${data_config}") "${_combine_dir}/wav.scp" \
                            > "${_combine_dir}/logdir/shuffled.wav.scp"
                        head -n "${speechlm_train_size}" "${_combine_dir}/logdir/shuffled.wav.scp" \
                            | cut -d' ' -f1 | sort > "${_combine_dir}/logdir/subsample_utts"
                        utils/subset_data_dir.sh --utt-list "${_combine_dir}/logdir/subsample_utts" \
                            "${_combine_dir}" "data/${_dset}/speech/${task}"
                        rm -rf "${_combine_dir}"
                    fi
                else
                    utils/combine_data.sh  data/${_dset}/speech/${task} ${speech_dirs}
                fi
            fi

            # combine text
            if [ -n "${text_data_paths}" ]; then
                mkdir -p data/${_dset}/text/${task}
                echo "${_dset}: Combine text: "${text_data_paths}
                if [ "${task}" = "textlm" ] && [ "${_dset}" = "train" ] && [ -n "${textlm_train_size}" ]; then
                    _full_text="data/${_dset}/text/${task}/text.full"
                    for f in ${text_data_paths}; do
                        cat "${f}"
                    done>"${_full_text}"
                    _nline=$(<"${_full_text}" wc -l)
                    if [ "${_nline}" -le "${textlm_train_size}" ]; then
                        log "textlm train has ${_nline} lines, <= requested ${textlm_train_size}. Keeping all."
                        cp "${_full_text}" "data/${_dset}/text/${task}/text"
                    else
                        log "Subsampling textlm train: ${_nline} -> ${textlm_train_size} lines"
                        # Same SIGPIPE-under-pipefail concern as the speechlm case above:
                        # write shuf's output to a file before running head on it.
                        shuf --random-source=<(yes "textlm_${data_config}") "${_full_text}" \
                            > "data/${_dset}/text/${task}/shuffled.text.full"
                        head -n "${textlm_train_size}" "data/${_dset}/text/${task}/shuffled.text.full" \
                            | sort > "data/${_dset}/text/${task}/text"
                        rm -f "data/${_dset}/text/${task}/shuffled.text.full"
                    fi
                    rm -f "${_full_text}"
                else
                    for f in ${text_data_paths}; do
                        cat "${f}"
                    done>"data/${_dset}/text/${task}/text"
                fi
            fi
        done

    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
