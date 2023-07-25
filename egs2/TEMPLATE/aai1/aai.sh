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
skip_stages=         # Spicify the stage to be skipped
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true # Skip packing and uploading to zenodo
skip_upload_hf=true  # Skip uploading to hugging face stages.
eval_valid_set=false # Run decoding for the validation set
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.
post_process_local_data_opts= # The options given to local/data.sh for additional processing in stage 4.
auxiliary_data_tags= # the names of training data for auxiliary tasks

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw_copy       # Feature type (raw, raw_copy, fbank_pitch, or extracted).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.


# AAI model related
aai_task=aai   # AAI task mode. 'aai'.
aai_tag=       # Suffix to the result dir for aai model training.
aai_exp=       # Specify the directory path for AAI experiment.
               # If this option is specified, aai_tag is ignored.
aai_stats_dir= # Specify the directory path for AAI statistics.
aai_config=    # Config for aai model training.
aai_args=      # Arguments for aai model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in aai config.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_ref=1   # Number of references for training.
            # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
num_inf=    # Number of inferences output by the model
            # Note that if it is not specified, it will be the same as num_ref. Otherwise, it will be overwritten.
            # In MixIT, number of outputs is larger than that of references.
sot_aai=false   # Whether to use Serialized Output Training (SOT)

# Upload model related
hf_repo=

batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.

inference_aai_model=valid.acc.ave.pth # AAI model path for decoding.
                                      # e.g.
                                      # inference_aai_model=train.loss.best.pth
                                      # inference_aai_model=3epoch.pth
                                      # inference_aai_model=valid.acc.best.pth
                                      # inference_aai_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
hyp_cleaner=none # Text cleaner for hypotheses (may be used with external tokenizers)
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
aai_speech_fold_length=800 # fold_length for speech data during AAI training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_stages    # Spicify the stage to be skipped (default="${skip_stages}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --skip_upload_hf    # Skip packing and uploading stages (default="${skip_upload_hf}").
    --eval_valid_set # Run decoding for the validation set (default="${eval_valid_set}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, raw_copy, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw or raw_copy, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # AAI model related
    --aai_task         # AAI task mode. Either 'aai' or 'aai_transducer'. (default="${aai_task}").
    --aai_tag          # Suffix to the result dir for aai model training (default="${aai_tag}").
    --aai_exp          # Specify the directory path for AAI experiment.
                       # If this option is specified, aai_tag is ignored (default="${aai_exp}").
    --aai_stats_dir    # Specify the directory path for AAI statistics (default="${aai_stats_dir}").
    --aai_config       # Config for aai model training (default="${aai_config}").
    --aai_args         # Arguments for aai model training (default="${aai_args}").
                       # e.g., --aai_args "--max_epoch 10"
                       # Note that it will overwrite args in aai config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_ref    # Number of references for training (default="${num_ref}").
                 # In supervised learning based speech recognition, it is equivalent to number of speakers.
    --num_inf    # Number of inference audio generated by the model (default="${num_inf}")
                 # Note that if it is not specified, it will be the same as num_ref. Otherwise, it will be overwritten.
    --sot_aai    # Whether to use Serialized Output Training (SOT) (default="${sot_aai}")

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_aai_model # AAI model path for decoding (default="${inference_aai_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").


    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --aai_speech_fold_length # fold_length for speech data during AAI training (default="${aai_speech_fold_length}").
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


# Check required arguments
if ! "${skip_train}"; then
    [ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
    [ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
fi
if ! "${eval_valid_set}"; then
    [ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };
else
    [ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
fi

if [ -n "${train_set}" ] && [ "${train_set}" = "${valid_set}" ]; then
    log "Error: train_set and valid_set must be different. --train_set ${train_set} --valid_set ${valid_set}"
    exit 1
fi

_test_sets=
for dset in ${test_sets}; do
    if [ "${dset}" = "${train_set}" ]; then
        log "Error: train_set and test_sets must be different. --train_set ${train_set} --test_sets ${test_sets}"
        exit 1
    fi
    if [ "${dset}" = "${valid_set}" ]; then
        log "Info: The valid_set '${valid_set}' is included in the test_sets. '--eval_valid_set true' is set and '${valid_set}' is removed from the test_sets"
        eval_valid_set=true
    elif [[ " ${_test_sets} " =~ [[:space:]]${dset}[[:space:]] ]]; then
        log "Info: ${dset} is duplicated in the test_sets. One is removed"
    else
        _test_sets+="${dset} "
    fi
done
test_sets=${_test_sets}

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = raw_copy ]; then
    # raw_copy is as same as raw except for skipping the format_wav stage
    data_feats=${dumpdir}/raw_copy
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

num_inf=${num_inf:=${num_ref}}
# Preprocessor related
if [ ${num_ref} -eq 1 ]; then
    # For single speaker, text file path and name are text
    ref_text_files_str="text "
    ref_text_names_str="text "
else
    # For multiple speakers, text file path and name are text_spk[1-N] and [text, text_spk2, ...]
    #TODO(simpleoier): later to support flexibly defined text prefix
    ref_text_files_str="text_spk1 "
    ref_text_names_str="text "
    for n in $(seq 2 ${num_ref}); do
        ref_text_files_str+="text_spk${n} "
        ref_text_names_str+="text_spk${n} "
    done
fi

# Set tag for naming of model directory
if [ -z "${aai_tag}" ]; then
    if [ -n "${aai_config}" ]; then
        aai_tag="$(basename "${aai_config}" .yaml)_${feats_type}"
    else
        aai_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${aai_args}" ]; then
        aai_tag+="$(echo "${aai_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        aai_tag+="_sp"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${aai_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        aai_stats_dir="${expdir}/aai_stats_${feats_type}_${lang}"
    else
        aai_stats_dir="${expdir}/aai_stats_${feats_type}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        aai_stats_dir+="_sp"
    fi
fi

# The directory used for training commands
if [ -z "${aai_exp}" ]; then
    aai_exp="${expdir}/aai_${aai_tag}"
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
    inference_tag+="_aai_model_$(echo "${inference_aai_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

fi

if "${skip_data_prep}"; then
    skip_stages+="1 2 3 4 5 "
fi
if "${skip_train}"; then
    skip_stages+="2 4 5 6 7 8 9 10 11 "
fi
if "${skip_eval}"; then
    skip_stages+="12 13 "
fi

if "${skip_upload}" && "${skip_upload_hf}"; then
    skip_stages+="14 15 16 "
elif "${skip_upload}"; then
    skip_stages+="15 "
elif "${skip_upload_hf}"; then
    skip_stages+="16 "
fi
skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"

# ========================== Main stages start from here. ==========================



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    if [ -n "${speed_perturb_factors}" ]; then
       log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
       for factor in ${speed_perturb_factors}; do
           if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
               scripts/utils/perturb_data_dir_speed.sh \
                   ${ref_text_files_str:+--utt_extra_files "${ref_text_files_str}"} \
                   "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
               _dirs+="data/${train_set}_sp${factor} "
           else
               # If speed factor is 1, same as the original
               _dirs+="data/${train_set} "
           fi
        done
        utils/combine_data.sh \
            ${ref_text_files_str:+--extra_files "${ref_text_files_str}"} \
            "data/${train_set}_sp" ${_dirs}
    else
       log "Skip stage 2: Speed perturbation"
    fi
fi

if [ -n "${speed_perturb_factors}" ]; then
    train_set="${train_set}_sp"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    if "${skip_train}"; then
        if "${eval_valid_set}"; then
            _dsets="${valid_set} ${test_sets}"
        else
            _dsets="${test_sets}"
        fi
    else
        _dsets="${train_set} ${valid_set} ${test_sets}"
    fi
    if [ "${feats_type}" = raw ]; then
        log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}

            # Copy reference text files if there is more than 1 reference


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
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_feats}${_suf}/${dset}/audio_format"
            else
                echo "${audio_format}" > "${data_feats}${_suf}/${dset}/audio_format"
            fi
        done

    elif [ "${feats_type}" = raw_copy ]; then
        # If you guaranteed that the data already satisfy the raw format, you can skip format_wav_scp.py for reduce the overhead
        for dset in ${_dsets}; do
            if [ -e "data/${dset}/segments" ]; then
                log "Error: data/${dset}/segments is existing. Please use --feats_type raw"
                exit 1
            fi
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"

                if [ -e "data/${dset}/utt2dur" ]; then
                    _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                    <data/${dset}/utt2dur awk '{ print $1, int($2*'${_fs}'); }' > "${data_feats}${_suf}/${dset}"/utt2num_samples

                elif [ -e "data/${dset}/utt2num_samples" ]; then
                    cp "data/${dset}/utt2num_samples" "${data_feats}${_suf}/${dset}"/utt2num_samples

                else
                    log "Error: data/${dset}/utt2dur or data/${dset}/utt2num_samples must be existing for train_set and valid_set. Please use --feats_type raw. If you'd like to perform this script for evaluation, please give --skip_train true"
                    exit 1
                fi
            fi

            # Copy reference text files if there is more than 1 reference
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                # shellcheck disable=SC2068
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${data_feats}${_suf}/${dset}
                done
            fi

            echo "raw" > "${data_feats}${_suf}/${dset}/feats_type"
            if "${multi_columns_input_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_feats}${_suf}/${dset}/audio_format"
            else
                echo "${audio_format}" > "${data_feats}${_suf}/${dset}/audio_format"
            fi
        done

    elif [ "${feats_type}" = fbank_pitch ]; then
        log "[Require Kaldi] Stage 3: ${feats_type} extract: data/ -> ${data_feats}"

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            # 1. Copy datadir
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

            # Copy reference text files if there is more than 1 reference
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                # shellcheck disable=SC2068
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${data_feats}${_suf}/${dset}
                done
            fi

            # 2. Feature extract
            _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
            steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
            utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

            # 3. Derive the the frame length and feature dimension
            scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

            # 4. Write feats_dim
            head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' \
                | cut -d, -f2 > ${data_feats}${_suf}/${dset}/feats_dim

            # 5. Write feats_type
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done

    elif [ "${feats_type}" = fbank ]; then
        log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}"
        log "${feats_type} is not supported yet."
        exit 1

    elif  [ "${feats_type}" = extracted ]; then
        log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}"
        # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            # Generate dummy wav.scp to avoid error by copy_data_dir.sh
            if [ ! -f data/"${dset}"/wav.scp ]; then
                if [ ! -f data/"${dset}"/segments ]; then
                    <data/"${dset}"/feats.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp
                else
                    <data/"${dset}"/segments awk ' { print($2,"<DUMMY>") }' > data/"${dset}"/wav.scp
                fi
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

            # Copy reference text files if there is more than 1 reference
            # shellcheck disable=SC2068
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${data_feats}${_suf}/${dset}
                done
            fi

            # Derive the the frame length and feature dimension
            _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
            scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

            pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats.scp |" - | \
                awk '{ print $2 }' | cut -d, -f2 > "${data_feats}${_suf}/${dset}/feats_dim"

            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done

    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

    # NOTE(kamo): Not applying to test_sets to keep original data
    for dset in "${train_set}" "${valid_set}"; do

        # Copy data dir
        utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
        cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

        # Remove short utterances
        _feats_type="$(<${data_feats}/${dset}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
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
        else
            # Get frame shift in ms from conf/fbank.conf
            _frame_shift=
            if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                # Assume using conf/fbank.conf for feature extraction
                _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
            fi
            if [ -z "${_frame_shift}" ]; then
                # If not existing, use the default number in Kaldi (=10ms).
                # If you are using different number, you have to change the following value manually.
                _frame_shift=10
            fi

            _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

            cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
            <"${data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                    >"${data_feats}/${dset}/feats_shape"
            <"${data_feats}/org/${dset}/feats.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/feats_shape"  \
                >"${data_feats}/${dset}/feats.scp"
        fi

        # Remove empty text
        # shellcheck disable=SC2068
        for ref_txt in ${ref_text_files[@]}; do
            <"${data_feats}/org/${dset}/${ref_txt}" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/${ref_txt}"
        done

        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh \
            ${ref_text_files_str:+--utt_extra_files "${ref_text_files_str}"} \
            "${data_feats}/${dset}"
    done

    if [ -n "${post_process_local_data_opts}" ]; then
        # Do any additional local data post-processing here
        local/data.sh ${post_process_local_data_opts} --aai_data_dir "${data_feats}/${train_set}"
    fi

    # shellcheck disable=SC2002,SC2068,SC2005
    for lm_txt in ${lm_train_text[@]}; do
        suffix=$(echo "$(basename ${lm_txt})" | sed 's/text//')
        <${lm_txt} awk -v suffix=${suffix} ' { if( NF != 1 ) {$1=$1 suffix; print $0; }} '
    done > "${data_feats}/lm_train.txt"
fi


# ========================== Data preparation is done here. ==========================

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    _aai_train_dir="${data_feats}/${train_set}"
    _aai_valid_dir="${data_feats}/${valid_set}"
    log "Stage 10: AAI collect stats: train_set=${_aai_train_dir}, valid_set=${_aai_valid_dir}"

    _opts=
    if [ -n "${aai_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.aai_train --print_config --optim adam
        _opts+="--config ${aai_config} "
    fi

    _feats_type="$(<${_aai_train_dir}/feats_type)"
    _audio_format="$(cat ${_aai_train_dir}/audio_format 2>/dev/null || echo ${audio_format})"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        if [[ "${_audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi
        _opts+="--frontend_conf fs=${fs} "
    else
        _scp=feats.scp
        _type=kaldi_ark
        _input_size="$(<${_aai_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "
    fi

    # 1. Split the key file
    _logdir="${aai_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${_aai_train_dir}/${_scp} wc -l)" "$(<${_aai_valid_dir}/${_scp} wc -l)")

    key_file="${_aai_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_aai_valid_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${aai_stats_dir}/run.sh'. You can resume the process from stage 10 using this script"
    mkdir -p "${aai_stats_dir}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${aai_stats_dir}/run.sh"; chmod +x "${aai_stats_dir}/run.sh"

    # 3. Submit jobs
    log "AAI collect-stats started... log: '${_logdir}/stats.*.log'"

    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.

    _opts+="--train_data_path_and_name_and_type ${_aai_train_dir}/${_scp},speech,${_type} "
    _opts+="--train_data_path_and_name_and_type ${_aai_train_dir}/${_scp},ema,${_type} "
    _opts+="--valid_data_path_and_name_and_type ${_aai_valid_dir}/${_scp},speech,${_type} "
    _opts+="--valid_data_path_and_name_and_type ${_aai_valid_dir}/${_scp},ema,${_type} "
    # shellcheck disable=SC2068

    echo $_opts

    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.${aai_task}_train \
            --collect_stats true \
            --use_preprocessor true \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${aai_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    if [ "${feats_normalize}" != global_mvn ]; then
        # Skip summerizaing stats if not using global MVN
        _opts+="--skip_sum_stats"
    fi
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${aai_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    # shellcheck disable=SC2068
    for ref_txt in ${ref_text_names[@]}; do
        <"${aai_stats_dir}/train/${ref_txt}_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${aai_stats_dir}/train/${ref_txt}_shape.${token_type}"

        <"${aai_stats_dir}/valid/${ref_txt}_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${aai_stats_dir}/valid/${ref_txt}_shape.${token_type}"
    done
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ] && ! [[ " ${skip_stages} " =~ [[:space:]]11[[:space:]] ]]; then
    _aai_train_dir="${data_feats}/${train_set}"
    _aai_valid_dir="${data_feats}/${valid_set}"
    log "Stage 11: AAI Training: train_set=${_aai_train_dir}, valid_set=${_aai_valid_dir}"

    _opts=
    if [ -n "${aai_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.aai_train --print_config --optim adam
        _opts+="--config ${aai_config} "
    fi

    _feats_type="$(<${_aai_train_dir}/feats_type)"
    _audio_format="$(cat ${_aai_train_dir}/audio_format 2>/dev/null || echo ${audio_format})"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        _ema=text
        # "sound" supports "wav", "flac", etc.
        if [[ "${_audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        elif [[ "${_audio_format}" == *multi* ]]; then
            _type=multi_columns_sound
        else
            _type=sound
        fi
        _fold_length="$((aai_speech_fold_length * 100))"
        _opts+="--frontend_conf fs=${fs} "
    else
        _scp=feats.scp
        _type=kaldi_ark
        _fold_length="${aai_speech_fold_length}"
        _input_size="$(<${_aai_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "

    fi
    if [ "${feats_normalize}" = global_mvn ]; then
        # Default normalization is utterance_mvn and changes to global_mvn
        _opts+="--normalize=global_mvn --normalize_conf stats_file=${aai_stats_dir}/train/feats_stats.npz "
    fi

    
    _opts+="--train_data_path_and_name_and_type ${_aai_train_dir}/${_scp},speech,${_type} "
    _opts+="--train_data_path_and_name_and_type ${_aai_train_dir}/${_ema},ema,pth "
    _opts+="--train_shape_file ${aai_stats_dir}/train/speech_shape "

    _opts+="--valid_data_path_and_name_and_type ${_aai_valid_dir}/${_scp},speech,${_type} "
    _opts+="--valid_data_path_and_name_and_type ${_aai_valid_dir}/${_ema},ema,pth "
    _opts+="--valid_shape_file ${aai_stats_dir}/train/speech_shape "


    log "Generate '${aai_exp}/run.sh'. You can resume the process from stage 11 using this script"
    mkdir -p "${aai_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${aai_exp}/run.sh"; chmod +x "${aai_exp}/run.sh"

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
    log "AAI training started... log: '${aai_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${aai_exp})"
    else
        jobname="${aai_exp}/train.log"
    fi

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${aai_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${aai_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.${aai_task}_train \
            --use_preprocessor true \
            --resume true \
            ${pretrained_model:+--init_param $pretrained_model} \
            --ignore_init_mismatch ${ignore_init_mismatch} \
            --fold_length "${_fold_length}" \
            --output_dir "${aai_exp}" \
            ${_opts} ${aai_args}

fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    aai_exp="${expdir}/${download_model}"
    mkdir -p "${aai_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${aai_exp}/config.txt"

    # Get the path of each file
    _aai_model_file=$(<"${aai_exp}/config.txt" sed -e "s/.*'aai_model_file': '\([^']*\)'.*$/\1/")
    _aai_train_config=$(<"${aai_exp}/config.txt" sed -e "s/.*'aai_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_aai_model_file}" "${aai_exp}"
    ln -sf "${_aai_train_config}" "${aai_exp}"
    inference_aai_model=$(basename "${_aai_model_file}")

    if [ "$(<${aai_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${aai_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${aai_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ] && ! [[ " ${skip_stages} " =~ [[:space:]]12[[:space:]] ]]; then
    log "Stage 12: Decoding: training_dir=${aai_exp}"

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
    if "${use_lm}"; then
        if "${use_word_lm}"; then
            _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
            _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
        else
            _opts+="--lm_train_config ${lm_exp}/config.yaml "
            _opts+="--lm_file ${lm_exp}/${inference_lm} "
        fi
    fi
    if "${use_ngram}"; then
         _opts+="--ngram_file ${ngram_exp}/${inference_ngram}"
    fi

    # 2. Generate run.sh
    log "Generate '${aai_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
    mkdir -p "${aai_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${aai_exp}/${inference_tag}/run.sh"; chmod +x "${aai_exp}/${inference_tag}/run.sh"

    inference_bin_tag=""
    if [ ${aai_task} == "aai" ]; then
        if "${use_k2}"; then
            # Now only _nj=1 is verified if using k2
            inference_bin_tag="_k2"

            _opts+="--is_ctc_decoding ${k2_ctc_decoding} "
            _opts+="--use_nbest_rescoring ${use_nbest_rescoring} "
            _opts+="--num_paths ${num_paths} "
            _opts+="--nll_batch_size ${nll_batch_size} "
            _opts+="--k2_config ${k2_config} "
        elif "${use_streaming}"; then
            inference_bin_tag="_streaming"
        elif "${use_maskctc}"; then
            inference_bin_tag="_maskctc"
        fi
    fi

    if "${eval_valid_set}"; then
        _dsets="org/${valid_set} ${test_sets}"
    else
        _dsets="${test_sets}"
    fi
    for dset in ${_dsets}; do
        _data="${data_feats}/${dset}"
        _dir="${aai_exp}/${inference_tag}/${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        _feats_type="$(<${_data}/feats_type)"
        _audio_format="$(cat ${_data}/audio_format 2>/dev/null || echo ${audio_format})"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            elif [[ "${_audio_format}" == *multi* ]]; then
                _type=multi_columns_sound
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
        if "${use_k2}"; then
          # Now only _nj=1 is verified if using k2
          _nj=1
        else
          _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
        fi

        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/aai_inference.*.log'"
        rm -f "${_logdir}/*.log"
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/aai_inference.JOB.log \
            ${python} -m espnet2.bin.${aai_task}_inference${inference_bin_tag} \
                --batch_size ${batch_size} \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --aai_train_config "${aai_exp}"/config.yaml \
                --aai_model_file "${aai_exp}"/"${inference_aai_model}" \
                --output_dir "${_logdir}"/output.JOB \
                ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/aai_inference.*.log) ; exit 1; }

        # 3. Calculate and report RTF based on decoding logs
        if [ ${aai_task} == "aai" ] && [ -z ${inference_bin_tag} ]; then
            log "Calculating RTF & latency... log: '${_logdir}/calculate_rtf.log'"
            rm -f "${_logdir}"/calculate_rtf.log
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _sample_shift=$(python3 -c "print(1 / ${_fs} * 1000)") # in ms
            ${_cmd} JOB=1 "${_logdir}"/calculate_rtf.log \
                pyscripts/utils/calculate_rtf.py \
                    --log-dir ${_logdir} \
                    --log-name "aai_inference" \
                    --input-shift ${_sample_shift} \
                    --start-times-marker "speech length" \
                    --end-times-marker "best hypo" \
                    --inf-num ${num_inf} || { cat "${_logdir}"/calculate_rtf.log; exit 1; }
        fi

        # 4. Concatenates the output files from each jobs
        # shellcheck disable=SC2068
        for ref_txt in ${ref_text_files[@]}; do
            suffix=$(echo ${ref_txt} | sed 's/text//')
            for f in token token_int score text; do
                if [ -f "${_logdir}/output.1/1best_recog/${f}${suffix}" ]; then
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1best_recog/${f}${suffix}"
                    done | sort -k1 >"${_dir}/${f}${suffix}"
                fi
            done
        done

    done
fi


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ] && ! [[ " ${skip_stages} " =~ [[:space:]]13[[:space:]] ]]; then
    log "Stage 13: Scoring"
    if [ "${token_type}" = phn ]; then
        log "Error: Not implemented for token_type=phn"
        exit 1
    fi

    if "${eval_valid_set}"; then
        _dsets="org/${valid_set} ${test_sets}"
    else
        _dsets="${test_sets}"
    fi
    for dset in ${_dsets}; do
        _data="${data_feats}/${dset}"
        _dir="${aai_exp}/${inference_tag}/${dset}"

        for _tok_type in "char" "word" "bpe"; do
            [ "${_tok_type}" = bpe ] && [ ! -f "${bpemodel}" ] && continue

            _opts="--token_type ${_tok_type} "
            if [ "${_tok_type}" = "char" ] || [ "${_tok_type}" = "word" ]; then
                _type="${_tok_type:0:1}er"
                _opts+="--non_linguistic_symbols ${nlsyms_txt} "
                _opts+="--remove_non_linguistic_symbols true "

            elif [ "${_tok_type}" = "bpe" ]; then
                _type="ter"
                _opts+="--bpemodel ${bpemodel} "

            else
                log "Error: unsupported token type ${_tok_type}"
            fi

            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            # shellcheck disable=SC2068
            for ref_txt in ${ref_text_files[@]}; do
                # Note(simpleoier): to get the suffix after text, e.g. "text_spk1" -> "_spk1"
                suffix=$(echo ${ref_txt} | sed 's/text//')

                # Tokenize text to ${_tok_type} level
                paste \
                    <(<"${_data}/${ref_txt}" \
                        ${python} -m espnet2.bin.tokenize_text  \
                            -f 2- --input - --output - \
                            --cleaner "${cleaner}" \
                            ${_opts} \
                            ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref${suffix:-${suffix}}.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/${ref_txt}"  \
                        ${python} -m espnet2.bin.tokenize_text  \
                            -f 2- --input - --output - \
                            ${_opts} \
                            --cleaner "${hyp_cleaner}" \
                            ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp${suffix:-${suffix}}.trn"

            done

            # Note(simpleoier): score across all possible permutations
            if [ ${num_ref} -gt 1 ] && [ -n "${suffix}" ]; then
                for i in $(seq ${num_ref}); do
                    for j in $(seq ${num_inf}); do
                        sclite \
                            ${score_opts} \
                            -r "${_scoredir}/ref_spk${i}.trn" trn \
                            -h "${_scoredir}/hyp_spk${j}.trn" trn \
                            -i rm -o all stdout > "${_scoredir}/result_r${i}h${j}.txt"
                    done
                done
                # Generate the oracle permutation hyp.trn and ref.trn
                pyscripts/utils/eval_perm_free_error.py --num-spkrs ${num_ref} \
                    --results-dir ${_scoredir}
            fi

            sclite \
                ${score_opts} \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout > "${_scoredir}/result.txt"

            log "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done
    done

    [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${aai_exp}"

    # Show results in Markdown syntax
    scripts/utils/show_aai_result.sh "${aai_exp}" > "${aai_exp}"/RESULTS.md
    cat "${aai_exp}"/RESULTS.md

fi


packed_model="${aai_exp}/${aai_exp##*/}_${inference_aai_model%.*}.zip"
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ] && ! [[ " ${skip_stages} " =~ [[:space:]]14[[:space:]] ]]; then
    log "Stage 14: Pack model: ${packed_model}"

    _opts=
    if "${use_lm}"; then
        _opts+="--lm_train_config ${lm_exp}/config.yaml "
        _opts+="--lm_file ${lm_exp}/${inference_lm} "
        _opts+="--option ${lm_exp}/perplexity_test/ppl "
        _opts+="--option ${lm_exp}/images "
    fi
    if [ "${feats_normalize}" = global_mvn ]; then
        _opts+="--option ${aai_stats_dir}/train/feats_stats.npz "
    fi
    if [ "${token_type}" = bpe ]; then
        _opts+="--option ${bpemodel} "
    fi
    if [ "${nlsyms_txt}" != none ]; then
        _opts+="--option ${nlsyms_txt} "
    fi
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack aai \
        --aai_train_config "${aai_exp}"/config.yaml \
        --aai_model_file "${aai_exp}"/"${inference_aai_model}" \
        ${_opts} \
        --option "${aai_exp}"/RESULTS.md \
        --option "${aai_exp}"/images \
        --outpath "${packed_model}"
fi


if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ] && ! [[ " ${skip_stages} " =~ [[:space:]]15[[:space:]] ]]; then
    log "Stage 15: Upload model to Zenodo: ${packed_model}"
    log "Warning: Upload model to Zenodo will be deprecated. We encourage to use Hugging Face"

    # To upload your model, you need to do:
    #   1. Sign up to Zenodo: https://zenodo.org/
    #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
    #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="
git checkout $(git show -s --format=%H)"

    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/aai1/ -> foo/aai1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/aai1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # Generate description file
    cat << EOF > "${aai_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${aai_exp}"/RESULTS.md)</code></pre></li>
<li><strong>AAI config</strong><pre><code>$(cat "${aai_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

    # NOTE(kamo): The model file is uploaded here, but not published yet.
    #   Please confirm your record at Zenodo and publish it by yourself.

    # shellcheck disable=SC2086
    espnet_model_zoo_upload \
        --file "${packed_model}" \
        --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
        --description_file "${aai_exp}"/description \
        --creator_name "${_creator_name}" \
        --license "CC-BY-4.0" \
        --use_sandbox false \
        --publish false
fi


if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ] && ! [[ " ${skip_stages} " =~ [[:space:]]16[[:space:]] ]]; then
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1
    log "Stage 16: Upload model to HuggingFace: ${hf_repo}"

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
    # /some/where/espnet/egs2/foo/aai1/ -> foo/aai1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/aai1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=automatic-speech-recognition
    # shellcheck disable=SC2034
    espnet_task=AAI
    # shellcheck disable=SC2034
    task_exp=${aai_exp}
    eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

    this_folder=${PWD}
    cd ${dir_repo}
    if [ -n "$(git status --porcelain)" ]; then
        git add .
        git commit -m "Update model"
    fi
    git push
    cd ${this_folder}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
