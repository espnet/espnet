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
pretrain_start_iter= # Pretrain starts from the specified iteration (0 mean MFCC iteraion)
pretrain_stop_iter=  # Pretrain is stopped from the specified iteration (0 mean MFCC iteraion)
skip_data_prep=false # Skip data preparation stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
pretrain_ngpu=      # The number of gpus in pretrain stage ("0" uses cpu, otherwise use gpu).
pretrain_num_nodes=1 # The number of nodes in pretrain stage.
finetune_ngpu=      # The number of gpus in finetune stage("0" uses cpu, otherwise use gpu).
finetune_num_nodes=1 # The number of nodes in finetune stage.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

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

# Pretrain related
pretrain_train_set=
pretrain_valid_set=
pretrain_config_list=
n_clusters_list=
feature_list=

# Finetuen related
finetune_train_set=
finetune_valid_set=
finetune_test_sets=
finetune_config=


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
Usage: $0 --finetune_train-set "<finetune_train_set_name>" --finetune_valid-set "<finetune_valid_set_name>" --finetune_test_sets "<finetune_test_set_names>"

Options:
    # General configuration
    --stage                # Processes starts from the specified stage (default="${stage}").
    --stop_stage     	   # Processes is stopped at the specified stage (default="${stop_stage}").
    --pretrain_start_iter  # Pretrain starts from the specified iteration (0 mean MFCC iteraion)
    --pretrain_stop_iter   # Pretrain is stopped from the specified iteration (0 mean MFCC iteraion)
    --skip_data_prep 	   # Skip data preparation stages (default="${skip_data_prep}").
    --skip_eval      	   # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    	   # Skip packing and uploading stages (default="${skip_upload}").
    --pretrain_ngpu=1      # The number of gpus in pretrain stage("0" uses cpu, otherwise use gpu).
    --pretrain_num_nodes=1 # The number of nodes in pretrain stage.    
    --finetune_ngpu=1      # The number of gpus in finetune stage("0" uses cpu, otherwise use gpu).
    --finetune_num_nodes=1 # The number of nodes in finetune stage.
    --nj             	   # The number of parallel jobs (default="${nj}").
    --inference_nj   	   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  	   # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        	   # Directory to dump features (default="${dumpdir}").
    --expdir         	   # Directory to save experiments (default="${expdir}").
    --python         	   # Specify python to execute espnet commands (default="${python}").
    # Data preparation related
    --local_data_opts	   # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type	   # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     	   # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               	   # Sampling rate (default="${fs}").
    --min_wav_duration 	   # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration 	   # Maximum duration in second (default="${max_wav_duration}").

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
    --pretrain_train_set     # Name of pretraining train set
    --pretrain_valid_set     # Name of pretraining valid set
    --finetune_train_set     # Name of training set (required).
    --finetune_valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --finetune_test_sets     # Name of test sets used for evaluating network training (required).
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
[ -z "${finetune_train_set}" ] && { log "${help_message}"; log "Error: --finetune_train_set is required"; exit 2; };
[ -z "${finetune_valid_set}" ] && { log "${help_message}"; log "Error: --finetune_valid_set is required"; exit 2; };
[ -z "${finetune_test_sets}" ] && { log "${help_message}"; log "Error: --finetune_test_sets is required"; exit 2; };

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

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for Librispeech & Librilight"
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     
    for ((iter=${pretrain_start_iter}; iter<=${pretrain_stop_iter};iter++)); do
	
	train_set=$(eval "echo ${pretrain_train_set}_\${feature_list[${iter}]}_km\${n_clusters_list[${iter}]}")
	valid_set=$(eval "echo ${pretrain_valid_set}_\${feature_list[${iter}]}_km\${n_clusters_list[${iter}]}")
	
	feats_km="${feature_list[${iter}]}"
	n_clusters=${n_clusters_list[${iter}]}
	dictdir="./data/${feats_km}_km${n_clusters}_token_list_iter${iter}/word"

	if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
	    log "Stage 2.iter${iter}: Running ${n_clusters} cluster K-means on ${feats_km} feature."
	    
	    ./local/km.sh \
		--nclusters ${n_clusters} \
		--feature-type ${feats_km} \
		--datadir "./data" \
		--kmrootdir "./exp" \
		--dictdir ${dictdir}

	fi
	
	if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
	    if [ "${feats_type}" = raw ]; then
		log "Stage 3.iter${iter}: Copy wav.scp: data/ -> ${data_feats}"
		
		for dset in "${train_set}" "${valid_set}"; do
		    utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"
		    echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
		done
	    else
		log "Error: not supported: --feats_type ${feats_type}"
		exit 2
	    fi
	fi
	
	if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
	    log "Stage 4.iter${iter}: ${feats_km} pretrain model collect stats: \
	    	       train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"
	    
	    asr_config="${pretrain_config_list[${iter}]}"
	    _asr_train_dir="${data_feats}/${train_set}"
	    _asr_valid_dir="${data_feats}/${valid_set}"
	    if [ -z "${asr_stats_dir}" ]; then
		if [ "${lang}" != noinfo ]; then
		    asr_stats_dir="${expdir}/pretrain_iter${iter}_stats_${feats_type}_${lang}"
		else
		    asr_stats_dir="${expdir}/pretrain_iter${iter}_stats_${feats_type}"
		fi
	    fi
	    
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
	    log "Generate '${asr_stats_dir}/run.sh'. You can resume the process from stage4.iter${iter} using this script"
	    mkdir -p "${asr_stats_dir}"; echo "${run_args} --stage 4 \"\$@\"; exit \$?" > "${asr_stats_dir}/run.sh"; chmod +x "${asr_stats_dir}/run.sh"
	    
	    # 3. Submit jobs
	    log "Hubert pretrain collect-stats started... log: '${_logdir}/stats.*.log'"
	    
	    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
	    #       but it's used only for deciding the sample ids.
	    
	    # shellcheck disable=SC2086
	    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
			 ${python} -m espnet2.bin.hubert_train \
			 --collect_stats true \
			 --use_preprocessor true \
			 --normalize none \
			 --bpemodel none \
			 --token_type word \
			 --token_list "${dictdir}/tokens.txt" \
			 --non_linguistic_symbols none \
			 --cleaner "${cleaner}" \
			 --g2p "${g2p}" \
			 --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
			 --train_data_path_and_name_and_type "${_asr_train_dir}/text,text,text" \
			 --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
			 --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
			 --train_shape_file "${_logdir}/train.JOB.scp" \
			 --valid_shape_file "${_logdir}/valid.JOB.scp" \
			 --output_dir "${_logdir}/stats.JOB" \
			 ${_opts} || { cat "${_logdir}"/stats.1.log; exit 1; }
	    
	    # 4. Aggregate shape files
	    _opts=
	    for i in $(seq "${_nj}"); do
		_opts+="--input_dir ${_logdir}/stats.${i} "
	    done
	    # shellcheck disable=SC2086
	    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${asr_stats_dir}"
	    
	    # Append the num-tokens at the last dimensions. This is used for batch-bins count
	    <"${asr_stats_dir}/train/text_shape" \
             awk -v N="$(<${dictdir}/tokens.txt wc -l)" '{ print $0 "," N }' \
             >"${asr_stats_dir}/train/text_shape.word"
	    
	    <"${asr_stats_dir}/valid/text_shape" \
             awk -v N="$(<${dictdir}/tokens.txt wc -l)" '{ print $0 "," N }' \
             >"${asr_stats_dir}/valid/text_shape.word"
	fi
	
	if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

	    asr_config="${pretrain_config_list[${iter}]}"
	    _asr_train_dir="${data_feats}/${train_set}"
	    _asr_valid_dir="${data_feats}/${valid_set}"
	    
	    log "Stage 5.iter${iter}: Hubert Pretraining: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"
	
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
			  "${asr_stats_dir}/train/text_shape.word" \
			  --num_splits "${num_splits_asr}" \
			  --output_dir "${_split_dir}"
		    touch "${_split_dir}/.done"
		else
		    log "${_split_dir}/.done exists. Spliting is skipped"
		fi
		
		_opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
		_opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
		_opts+="--train_shape_file ${_split_dir}/speech_shape "
		_opts+="--train_shape_file ${_split_dir}/text_shape.word "
		_opts+="--multiple_iterator true "
		
	    else
		_opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/${_scp},speech,${_type} "
		_opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text,text,text "
		_opts+="--train_shape_file ${asr_stats_dir}/train/speech_shape "
		_opts+="--train_shape_file ${asr_stats_dir}/train/text_shape.word "
	    fi

	    if [ -z "${asr_tag}" ]; then
		if [ -n "${asr_config}" ]; then
		    asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
		else
		    asr_tag="train_${feats_type}"
		fi
	    fi
	    
	    if [ -z "${asr_exp}" ]; then
		asr_exp="${expdir}/pretrain_${asr_tag}_iter${iter}"
	    fi

	    log "Generate '${asr_exp}/run.sh'. You can resume the process from stage 5 using this script"
	    mkdir -p "${asr_exp}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${asr_exp}/run.sh"; chmod +x "${asr_exp}/run.sh"
	    
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
		      --ngpu "${pretrain_ngpu}" \
		      --num_nodes "${pretrain_num_nodes}" \
		      --init_file_prefix "${asr_exp}"/.dist_init_ \
		      --multiprocessing_distributed false -- \
		      ${python} -m espnet2.bin.hubert_train \
                      --use_preprocessor true \
		      --normalize null \
                      --bpemodel none \
                      --token_type word \
		      --token_list "${dictdir}/tokens.txt" \
                      --non_linguistic_symbols none \
                      --cleaner "${cleaner}" \
                      --g2p "${g2p}" \
                      --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                      --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                      --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
                      --valid_shape_file "${asr_stats_dir}/valid/text_shape.word" \
                      --resume true \
                      --fold_length "${_fold_length}" \
                      --fold_length "${asr_text_fold_length}" \
                      --output_dir "${asr_exp}" \
                      ${_opts}
	else
	    log "Skip the training stages"
	fi
    done
fi
set -x
if [ ${stop_stage} -ge 6 ]; then
    ./asr.sh \
	--stage "${stage}" --stop-stage "${stop_stage}" \
	--lang en \
	--ngpu "${finetune_ngpu}" \
	--num_nodes "${finetune_num_nodes}" \
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
	--token_type "${token_type}" \
	--lm_train_text "data/${finetune_train_set}/text" \
	--inference_asr_model valid.loss.ave.pth \
	--feats-normalize null  "$@"
    
fi

