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

stage=1
stop_stage=100

start_iter=0
stop_iter=2
pretrain_config_iter0=conf/tuning/train_asr_hubert_base_960h_full_pretrain_gpu32.yaml
pretrain_config_iter1=conf/tuning/train_asr_hubert_base_960h_full_pretrain_it1.yaml
pretrain_config_iter2=conf/tuning/train_asr_hubert_base_960h_full_pretrain_it2.yaml

lm_config=conf/tuning/train_lm_transformer2.yaml # didnt' use
inference_config=conf/decode_asr.yaml

finetune_train_set="train_10h"
finetune_valid_set="dev_other"
finetune_test_sets="dev_clean" #"test_clean test_other dev_clean dev_other"

finetune_asr_config=conf/tuning/train_asr_hubert_base_10h_finetuning.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

n_clusters_iter0=100
n_clusters_iter1=500
n_clusters_iter2=500
feature_iter0="mfcc"
feature_iter1="HuBERT6"
feature_iter2="HuBERT9"

for ((iter=${start_iter}; iter<=${stop_iter};iter++)); do
    pretrain_config_list[${iter}]=pretrain_config_iter${iter}
    n_clusters_list[${iter}]=n_clusters_iter${iter}
    feature_list[${iter}]=feature_list_iter${iter}
done

# Feature extraction related
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
skip_data_prep=false # Skip data preparation stages.
skip_km=false        # Skip k-means training stage.
skip_pretrain=false  # Skip Hubert pretraining stage.
skip_finetune=false  # Skip ASR finetune training stage.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
pretrain_ngpu=1      # The number of gpus in pretrain stage ("0" uses cpu, otherwise use gpu).
pretrain_num_nodes=1 # The number of nodes in pretrain stage.
finetune_ngpu=1      # The number of gpus in finetune stage("0" uses cpu, otherwise use gpu).
finetune_num_nodes=1 # The number of nodes in finetune stage.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=char      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
pad="<pad>"         # pad symbol
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Language model related
use_lm=true       # Use language model for ASR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the direcotry path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the direcotry path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ASR model related
asr_tag=       # Suffix to the result dir for asr model training.
asr_exp=       # Specify the direcotry path for ASR experiment.
               # If this option is specified, asr_tag is ignored.
asr_stats_dir= # Specify the direcotry path for ASR statistics.
asr_config=    # Config for asr model training.
asr_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in asr config.
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_asr=1           # Number of splitting for lm corpus.

# Decoding related
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language modle path for decoding.
inference_asr_model=valid.acc.best.pth # ASR model path for decoding.
                                      # e.g.
                                      # inference_asr_model=train.loss.best.pth
                                      # inference_asr_model=3epoch.pth
                                      # inference_asr_model=valid.acc.best.pth
                                      # inference_asr_model=valid.loss.ave.pth
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
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
asr_speech_fold_length=800 # fold_length for speech data during ASR training.
asr_text_fold_length=150   # fold_length for text data during ASR training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
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
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the direcotry path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the direcotry path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ASR model related
    --asr_tag          # Suffix to the result dir for asr model training (default="${asr_tag}").
    --asr_exp          # Specify the direcotry path for ASR experiment.
                       # If this option is specified, asr_tag is ignored (default="${asr_exp}").
    --asr_stats_dir    # Specify the direcotry path for ASR statistics (default="${asr_stats_dir}").
    --asr_config       # Config for asr model training (default="${asr_config}").
    --asr_args         # Arguments for asr model training (default="${asr_args}").
                       # e.g., --asr_args "--max_epoch 10"
                       # Note that it will overwrite args in asr config.
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_asr   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language modle path for decoding (default="${inference_lm}").
    --inference_asr_model # ASR model path for decoding (default="${inference_asr_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --asr_speech_fold_length # fold_length for speech data during ASR training (default="${asr_speech_fold_length}").
    --asr_text_fold_length   # fold_length for text data during ASR training (default="${asr_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
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

# Use the same text as ASR for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi


# Set tag for naming of model directory
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
    # Add overwritten arg's info
    if [ -n "${asr_args}" ]; then
        asr_tag+="$(echo "${asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_tag+="_sp"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    if [ "${lang}" != noinfo ]; then
        lm_tag+="_${lang}_${lm_token_type}"
    else
        lm_tag+="_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${asr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${lang}_${token_type}"
    else
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_stats_dir+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${lang}_${lm_token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${asr_exp}" ]; then
    asr_exp="${expdir}/asr_${asr_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
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
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${test_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for ((iter=${start_iter}; iter<=${stop_iter};iter++)); do
	train_set="train_960_${feature_list[${iter}]}_km${n_clusters_list[${iter}]}"
	valid_set="dev_${feature_mfcc}_km${mfcc_n_clusters}"

	echo ${train_set}
	echo ${valid_set}
	exit 0
	
	if ! "${skip_km}"; then
	    log "Stage 2.0: Running K-means on ${feature} feature."
	    feats_km=${features_list[${n_iter}]}
	    n_clusters=${n_clusters_list[${n_iter}]}
	    ./local/km.s \
		--nclusters ${n_clusters} \
		--feature-type ${feats_km} \
		--datadir "./data" \
		--kmrootdir "./exp" \
		--dictdir "./data/${feats_km}_km${n_clusters}_token_list/word"

	if [ "${feats_type}" = raw ]; then
	    log "Stage 2.1: Format wav.scp: data/ -> ${data_feats}"
	    for dset in "${train_set}" "${valid_set}"; do
		utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"
		rm -f ${data_feats}$/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
		_opts=
		if [ -e data/"${dset}"/segments ]; then
		    _opts+="--segments data/${dset}/segments "
		fi
		scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
						--audio-format "${audio_format}" --fs "${fs}" ${_opts} \
						"data/${dset}/wav.scp" "${data_feats}/${dset}"
		echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
	    done
	else
	    log "Error: not supported: --feats_type ${feats_type}"
	    exit 2
	fi
    fi
    ######################pretrain with mfcc label
    if ! "${skip_pretrain}"; then
	asr_config=${pretrain_configs}[${n_iteraions}]
	_asr_train_dir="${data_feats}/${train_set}"
	_asr_valid_dir="${data_feats}/${valid_set}"
	log "Stage 2.2: ${feature_mfcc} pretrain model collect stats: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"
    
	_opts=
	if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
	fi

	_feats_type="$(<${_asr_train_dir}/feats_type)"
	if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
		_type=kaldi_ark
            else
		# "sound" supports "wav", "flac", etc.
		_type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
	else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
	fi
    
	# 1. Split the key file
	_logdir="${asr_stats_dir}/logdir"
	mkdir -p "${_logdir}"
    
	# Get the minimum number among ${nj} and the number lines of input files
	_nj=$(min "${nj}" "$(<${_asr_train_dir}/${_scp} wc -l)" "$(<${_asr_valid_dir}/${_scp} wc -l)")
	
	key_file="${_asr_train_dir}/${_scp}"
	split_scps=""
	for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
	done
	# shellcheck disable=SC2086
	utils/split_scp.pl "${key_file}" ${split_scps}
	
	key_file="${_asr_valid_dir}/${_scp}"
	split_scps=""
	for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
	done
	# shellcheck disable=SC2086
	utils/split_scp.pl "${key_file}" ${split_scps}
    
	# 2. Generate run.sh
	log "Generate '${asr_stats_dir}/run.sh'. You can resume the process from stage 9 using this script"
	mkdir -p "${asr_stats_dir}"; echo "${run_args} --stage 9 \"\$@\"; exit \$?" > "${asr_stats_dir}/run.sh"; chmod +x "${asr_stats_dir}/run.sh"
    
	# 3. Submit jobs
	log "ASR collect-stats started... log: '${_logdir}/stats.*.log'"
	
	# NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
	#       but it's used only for deciding the sample ids.
	
	# shellcheck disable=SC2086
	${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
		     ${python} -m espnet2.bin.hubert_train \
                     --collect_stats true \
                     --use_preprocessor true \
                     --bpemodel "${bpemodel}" \
                     --token_type "${token_type}" \
                     --token_list "${token_list}" \
                     --non_linguistic_symbols "${nlsyms_txt}" \
                     --cleaner "${cleaner}" \
                     --g2p "${g2p}" \
                     --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
                     --train_data_path_and_name_and_type "${_asr_train_dir}/text,text,text" \
                     --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                     --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                     --train_shape_file "${_logdir}/train.JOB.scp" \
                     --valid_shape_file "${_logdir}/valid.JOB.scp" \
                     --output_dir "${_logdir}/stats.JOB" \
                     ${_opts} ${asr_args} || { cat "${_logdir}"/stats.1.log; exit 1; }
	
	# 4. Aggregate shape files
	_opts=
	for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
	done
	# shellcheck disable=SC2086
	${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${asr_stats_dir}"
	
	# Append the num-tokens at the last dimensions. This is used for batch-bins count
	<"${asr_stats_dir}/train/text_shape" \
         awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
         >"${asr_stats_dir}/train/text_shape.${token_type}"
	
	<"${asr_stats_dir}/valid/text_shape" \
         awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
         >"${asr_stats_dir}/valid/text_shape.${token_type}"


	# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
	asr_config=${pretrain_mfcc_config}
	_asr_train_dir="${data_feats}/${train_set}"
	_asr_valid_dir="${data_feats}/${valid_set}"
	log "Stage 5: MFCC Hubert Pretraining: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"
	
	_opts=
	if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
	fi
    
	_feats_type="$(<${_asr_train_dir}/feats_type)"
	if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
		_type=kaldi_ark
            else
		_type=sound
            fi
            _fold_length="$((asr_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
	else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${asr_speech_fold_length}"
            _input_size="$(<${_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
	
	fi
	if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz "
	fi
    
	if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.
	    
            _split_dir="${asr_stats_dir}/splits${num_splits_asr}"
            if [ ! -f "${_split_dir}/.done" ]; then
		rm -f "${_split_dir}/.done"
		${python} -m espnet2.bin.split_scps \
			  --scps \
			  "${_asr_train_dir}/${_scp}" \
			  "${_asr_train_dir}/text" \
			  "${asr_stats_dir}/train/speech_shape" \
			  "${asr_stats_dir}/train/text_shape.${token_type}" \
			  --num_splits "${num_splits_asr}" \
			  --output_dir "${_split_dir}"
		touch "${_split_dir}/.done"
            else
		log "${_split_dir}/.done exists. Spliting is skipped"
            fi
	
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "
	
	else
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text,text,text "
            _opts+="--train_shape_file ${asr_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${asr_stats_dir}/train/text_shape.${token_type} "
	fi

	log "Generate '${asr_exp}/run.sh'. You can resume the process from stage 10 using this script"
	mkdir -p "${asr_exp}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${asr_exp}/run.sh"; chmod +x "${asr_exp}/run.sh"
	
	# NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
	log "ASR training started... log: '${asr_exp}/train.log'"
	if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${asr_exp})"
	else
            jobname="${asr_exp}/train.log"
	fi

	# shellcheck disable=SC2086
	${python} -m espnet2.bin.launch \
		  --cmd "${cuda_cmd} --name ${jobname}" \
		  --log "${asr_exp}"/train.log \
		  --ngpu "${ngpu}" \
		  --num_nodes "${num_nodes}" \
		  --init_file_prefix "${asr_exp}"/.dist_init_ \
		  --multiprocessing_distributed false -- \
		  ${python} -m espnet2.bin.hubert_train \
                  --use_preprocessor true \
                  --bpemodel "${bpemodel}" \
                  --token_type "${token_type}" \
                  --token_list "${token_list}" \
                  --non_linguistic_symbols "${nlsyms_txt}" \
                  --cleaner "${cleaner}" \
                  --g2p "${g2p}" \
                  --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                  --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                  --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
                  --valid_shape_file "${asr_stats_dir}/valid/text_shape.${token_type}" \
                  --resume true \
                  --fold_length "${_fold_length}" \
                  --fold_length "${asr_text_fold_length}" \
                  --output_dir "${asr_exp}" \
                  ${_opts} ${asr_args}

    fi
else
    log "Skip the training stages"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Finetune Training: train_set=${finetune_train}, dev_set=${finetune_dev}"
    ./asr.sh \
	--lang en \
	--ngpu ${finetune_ngpu} \
	--nj ${nj} \
	--max_wav_duration 30 \
	--asr_config "${finetune_config}" \
	--use_lm ${use_lm} \
	--lm_config "${lm_config}" \
	--inference_config "${inference_config}" \
	--train_set "${finetune_train_set}" \
	--valid_set "${finetune_valid_set}" \
	--test_sets "${finetune_test_sets}" \
	--bpe_train_text "data/${finetune_train_set}/text" \
	--token_type char \
	--lm_train_text "data/${finetune_train_set}/text" \
	--inference_asr_model valid.loss.ave.pth \
	--feats-normalize null  "$@" 
fi

