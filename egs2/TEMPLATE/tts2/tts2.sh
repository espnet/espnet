#!/usr/bin/env bash

# Copyright 2019 Tomoki Hayashi
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
skip_upload_hf=true # Skip uploading to hugging face stages.
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
max_wav_duration=20        # Maximum duration in second.
use_sid=false              # Whether to use speaker id as the inputs (Need utt2spk in data directory).
use_lid=false              # Whether to use language id as the inputs (Need utt2lang in data directory).
fs=16000                   # Sampling rate.
n_fft=1024                 # The number of fft points.
n_shift=256                # The number of shift points.
win_length=null            # Window length.
fmin=80                    # Minimum frequency of Mel basis.
fmax=7600                  # Maximum frequency of Mel basis.
n_mels=80                  # The number of mel basis.
# Only used for the model using pitch & energy features (e.g. FastSpeech2)
f0min=80  # Maximum f0 for pitch extraction.
f0max=400 # Minimum f0 for pitch extraction.

# Speech Discretization Related
discrete_stage=1                # discretization stage
discrete_stop_stage=3           # discretization stop stage
clustering_nj=20                # Number of threads used for kmeans clustering
feature_dir="dump/feats"        # Feature directory for dumped feature
km_tag=                         # KMeans tagging
use_gpu_feat_extract=true       # Whether to use gpu for feature extraction
feature_layer=6                 # Layers for feature extraction
s3prl_upstream_name=hubert      # S3PRL upstream name for feature extraction
feature_clustering_tool="sklearn" # Tool for feature clustering (sklearn or faiss or cuml)
clustering_portion=0.1          # the portion of data used for clustering
feature_num_clusters=500        # Number of feature clusters
storage_save_mode=true          # Save storage on SSL feature extraction
                                # If true, feature extraction and kmeans clustering on the fly

# Speaker embedding related
use_spk_embed=false      # Whether to use speaker embedding.
spk_embed_tag=espnet_spk # The additional tag of speaker embedding folder, use "xvector" for compatibility.
spk_embed_gpu_inference=false # Whether to use gpu to inference speaker embedding.
spk_embed_tool=espnet    # Toolkit for extracting x-vector (speechbrain, rawnet, espnet, kaldi).
spk_embed_model=espnet/voxcelebs12_rawnet3  # For only espnet, speechbrain, or rawnet.

# Vocabulary related
oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol.
sos_eos="<sos/eos>" # sos and eos symbols.

# Training related
train_config=""    # Config for training.
train_args=""      # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
tts2_exp=""         # Specify the directory path for experiment. If this option is specified, tag is ignored.
tts2_stats_dir=""   # Specify the directory path for statistics. If empty, automatically decided.
num_splits=1       # Number of splitting for tts2 corpus.
teacher_dumpdir="" # Directory of teacher outputs (needed if tts2=fastspeech).
write_collected_feats=false # Whether to dump features in stats collection.
tts2_task=tts2                # Discrete TTS task (tts2 or gan_tts2).
write_collected_feats=true    # will be much faster if do so

# Decoding related
inference_config="" # Config for decoding.
inference_args=""   # Arguments for decoding (e.g., "--threshold 0.75").
                    # Note that it will overwrite args in inference config.
inference_tag=""    # Suffix for decoding directory.
inference_model=train.loss.ave.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
vocoder_file=none  # Vocoder parameter file, If set to none, Griffin-Lim will be used.
download_model=""  # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""     # Name of training set.
valid_set=""     # Name of validation set used for monitoring/tuning network training.
test_sets=""     # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
srctexts=""      # Texts to create src_token list. Multiple items can be specified.
nlsyms_txt=none  # Non-linguistic symbol list (needed if existing).
src_token_type=phn   # Transcription type (char or phn).
cleaner=tacotron # Text cleaner.
g2p=g2p_en       # g2p method (needed if src_token_type=phn).
lang=noinfo      # The language type of corpus.
text_fold_length=150   # fold_length for text data.
speech_fold_length=800 # fold_length for speech data.

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
    --use_spk_embed    # Whether to use speaker_embedding (default="${use_spk_embed}").
    --spk_embed_tag    # The tag of speaker embedding folder, use "xvector" for compatibility (default="${spk_embed_tag}").
    --spk_embed_gpu_inference # Whether to use gpu to inference speaker embedding (default="${spk_embed_gpu_inference}").
    --spk_embed_tool   # Toolkit for generating the speaker embedding (default="${spk_embed_tool}").
    --spk_embed_model  # Pretrained model to generate the speaker embedding (default="${spk_embed_model}").
    --use_sid          # Whether to use speaker id as the inputs (default="${use_sid}").
    --use_lid          # Whether to use language id as the inputs (default="${use_lid}").
    --fs               # Sampling rate (default="${fs}").
    --oov              # Out of vocabrary symbol (default="${oov}").
    --blank            # CTC blank symbol (default="${blank}").
    --sos_eos          # sos and eos symbole (default="${sos_eos}").

    # Discrete unit related
    --discrete_stage           # The start stage of clustering stage (default="${discrete_stage}").
    --discrete_stop_stage      # The stop stage of clustering stage (default="${discrete_stop_stage}").
    --clustering_nj              # Number of threads of clustering process (default="${clustering_nj}").
    --feature_dir              # Feature directory for dumped feature (default="${feature_dir}")
    --km_tag                   # KMeans tagging (default="${km_tag}")
    --use_gpu_feat_extract     # Whether to use gpu for feature extraction (default="${use_gpu_feat_extract}")
    --feature_layer            # Layers for feature extraction (default="${feature_layer}")
    --s3prl_upstream_name      # S3PRL upstream name for feature extraction (default="${s3prl_upstream_name}")
    --feature_clustering_tool  # Tool to do feature clustering (default="${feature_clustering_tool}")
    --clustering_portion       # The portion of data that is used for clustering (default="${clustering_portion}").
    --feature_num_clusters     # Number of clusters for feature clustering pooling (default="${feature_num_clusters}").
    --storage_save_mode=true   # Save storage on SSL feature extraction (default="${storage_save_mode}").
                               # If true, feature extraction and kmeans clustering on the fly

    # Training related
    --train_config  # Config for training (default="${train_config}").
    --train_args    # Arguments for training (default="${train_args}").
                    # e.g., --train_args "--max_epoch 1"
                    # Note that it will overwrite args in train config.
    --tag           # Suffix for training directory (default="${tag}").
    --tts2_exp       # Specify the directory path for experiment.
                    # If this option is specified, tag is ignored (default="${tts2_exp}").
    --tts2_stats_dir # Specify the directory path for statistics.
                    # If empty, automatically decided (default="${tts2_stats_dir}").
    --num_splits    # Number of splitting for tts2 corpus (default="${num_splits}").
    --teacher_dumpdir       # Directory of teacher outputs (needed if tts2=fastspeech, default="${teacher_dumpdir}").
    --write_collected_feats # Whether to dump features in statistics collection (default="${write_collected_feats}").
    --tts2_task              # Discrete TTS task {tts2 or gan_tts2} (default="${tts2_task}").

    # Decoding related
    --inference_config  # Config for decoding (default="${inference_config}").
    --inference_args    # Arguments for decoding, (default="${inference_args}").
                        # e.g., --inference_args "--threshold 0.75"
                        # Note that it will overwrite args in inference config.
    --inference_tag     # Suffix for decoding directory (default="${inference_tag}").
    --inference_model   # Model path for decoding (default=${inference_model}).
    --vocoder_file      # Vocoder paramemter file (default=${vocoder_file}).
                        # If set to none, Griffin-Lim vocoder will be used.
    --download_model    # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set          # Name of training set (required).
    --valid_set          # Name of validation set used for monitoring/tuning network training (required).
    --test_sets          # Names of test sets (required).
                         # Note that multiple items (e.g., both dev and eval sets) can be specified.
    --srctexts           # Texts to create src_token list (required).
                         # Note that multiple items can be specified.
    --nlsyms_txt         # Non-linguistic symbol list (default="${nlsyms_txt}").
    --src_token_type         # Transcription type (default="${src_token_type}").
    --cleaner            # Text cleaner (default="${cleaner}").
    --g2p                # g2p method (default="${g2p}").
    --lang               # The language type of corpus (default="${lang}").
    --text_fold_length   # Fold length for text data (default="${text_fold_length}").
    --speech_fold_length # Fold length for speech data (default="${speech_fold_length}").
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

# Check src_token list type
src_token_listdir="${dumpdir}/src_token_list/${src_token_type}"
if [ "${cleaner}" != none ]; then
    src_token_listdir+="_${cleaner}"
fi
if [ "${src_token_type}" = phn ]; then
    src_token_listdir+="_${g2p}"
fi
src_token_list="${src_token_listdir}/src_tokens.txt"

# Set tag for KMeans directory
if [ -z "${km_tag}" ]; then
    km_tag="${s3prl_upstream_name}_layer${feature_layer}_${feature_num_clusters}"
fi
km_dir="dump/${km_tag}"
tgt_token_list="${km_dir}/tgt_tokens.txt"

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)_${feats_type}_${src_token_type}"
    else
        tag="train_${feats_type}_${src_token_type}"
    fi
    if [ "${cleaner}" != none ]; then
        tag+="_${cleaner}"
    fi
    if [ "${src_token_type}" = phn ]; then
        tag+="_${g2p}"
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
if [ -z "${tts2_stats_dir}" ]; then
    tts2_stats_dir="${expdir}/tts2_stats_${feats_type}"
    tts2_stats_dir+="_${src_token_type}"
    if [ "${cleaner}" != none ]; then
        tts2_stats_dir+="_${cleaner}"
    fi
    if [ "${src_token_type}" = phn ]; then
        tts2_stats_dir+="_${g2p}"
    fi
fi
# The directory used for training commands
if [ -z "${tts2_exp}" ]; then
    tts2_exp="${expdir}/tts2_${tag}"
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
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                _opts+="--segments data/${dset}/segments "
            fi
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] ; then
        log "Stage 3: Prepare for additional token or embedding (e.g., spk, lang)."

        # Extract speaker embedding
        if "${use_spk_embed}"; then
            if [ "${spk_embed_tool}" = "kaldi" ]; then
                log "Stage 3.1: Extract X-vector with Kaldi: data/ -> ${dumpdir}/${spk_embed_tag} (Require Kaldi)"
                # Download X-vector pretrained model
                xvector_exp=${expdir}/xvector_nnet_1a
                if [ ! -e "${xvector_exp}" ]; then
                    log "X-vector model does not exist. Download pre-trained model."
                    wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
                    tar xvf 0008_sitw_v2_1a.tar.gz
                    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
                    mv 0008_sitw_v2_1a/exp/xvector_nnet_1a "${xvector_exp}"
                    rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
                fi

                # Generate the MFCC features, VAD decision, and X-vector
                for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                    if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                        _suf="/org"
                    else
                        _suf=""
                    fi
                    # 1. Copy datadir and resample to 16k
                    utils/copy_data_dir.sh "${data_feats}${_suf}/${dset}" "${dumpdir}/mfcc/${dset}"
                    utils/data/resample_data_dir.sh 16000 "${dumpdir}/mfcc/${dset}"

                    # 2. Extract mfcc features
                    _nj=$(min "${nj}" "$(<${dumpdir}/mfcc/${dset}/utt2spk wc -l)")
                    steps/make_mfcc.sh --nj "${_nj}" --cmd "${train_cmd}" \
                        --write-utt2num-frames true \
                        --mfcc-config conf/mfcc.conf \
                        "${dumpdir}/mfcc/${dset}"
                    utils/fix_data_dir.sh "${dumpdir}/mfcc/${dset}"

                    # 3. Compute VAD decision
                    _nj=$(min "${nj}" "$(<${dumpdir}/mfcc/${dset}/spk2utt wc -l)")
                    sid/compute_vad_decision.sh --nj ${_nj} --cmd "${train_cmd}" \
                        --vad-config conf/vad.conf \
                        "${dumpdir}/mfcc/${dset}"
                    utils/fix_data_dir.sh "${dumpdir}/mfcc/${dset}"

                    # 4. Extract X-vector
                    sid/nnet3/xvector/extract_xvectors.sh --nj "${_nj}" --cmd "${train_cmd}" \
                        "${xvector_exp}" \
                        "${dumpdir}/mfcc/${dset}" \
                        "${dumpdir}/${spk_embed_tag}/${dset}"

                    # 5. Filter scp
                    # NOTE(kan-bayashi): Since sometimes mfcc or x-vector extraction is failed,
                    #   the number of utts will be different from the original features (raw or fbank).
                    #   To avoid this mismatch, perform filtering of the original feature scp here.
                    cp "${data_feats}${_suf}/${dset}"/wav.{scp,scp.bak}
                    <"${data_feats}${_suf}/${dset}/wav.scp.bak" \
                        utils/filter_scp.pl "${dumpdir}/${spk_embed_tag}/${dset}/${spk_embed_tag}.scp" \
                        >"${data_feats}${_suf}/${dset}/wav.scp"
                    utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"
                done
            else
                # Assume that others toolkits are python-based
                log "Stage 3.1: Extract speaker embedding: data/ -> ${dumpdir}/${spk_embed_tag} using python toolkits"

                if ${spk_embed_gpu_inference}; then
                    _cmd="${cuda_cmd}"
                    _ngpu=1
                else
                    _cmd="${decode_cmd}"
                    _ngpu=0
                fi

                for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                    if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                        _suf="/org"
                    else
                        _suf=""
                    fi
                    if [ "${spk_embed_tool}" = "rawnet" ]; then
                        spk_embed_model="RawNet"
                    fi

                    ${_cmd} --gpu "${_ngpu}" ${dumpdir}/${spk_embed_tag}/${dset}/spk_embed_extract.log \
                    pyscripts/utils/extract_spk_embed.py \
                        --pretrained_model ${spk_embed_model} \
                        --toolkit ${spk_embed_tool} \
			--spk_embed_tag ${spk_embed_tag} \
                        ${data_feats}${_suf}/${dset} \
                        ${dumpdir}/${spk_embed_tag}/${dset}
                done
            fi
        else
            log "Skip Stage 3.1, no speaker embedding extraction set"
        fi

        # Prepare spk id input
        if "${use_sid}"; then
            log "Stage 3.2: Prepare speaker id: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                if [ "${dset}" = "${train_set}" ]; then
                    # Make spk2sid
                    # NOTE(kan-bayashi): 0 is reserved for unknown speakers
                    echo "<unk> 0" > "${data_feats}${_suf}/${dset}/spk2sid"
                    cut -f 2 -d " " "${data_feats}${_suf}/${dset}/utt2spk" | sort | uniq | \
                        awk '{print $1 " " NR}' >> "${data_feats}${_suf}/${dset}/spk2sid"
                fi
                pyscripts/utils/utt2spk_to_utt2sid.py \
                    "${data_feats}/org/${train_set}/spk2sid" \
                    "${data_feats}${_suf}/${dset}/utt2spk" \
                    > "${data_feats}${_suf}/${dset}/utt2sid"
            done
        fi

        # Prepare lang id input
        if "${use_lid}"; then
            log "Stage 3.3: Prepare lang id: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                if [ "${dset}" = "${train_set}" ]; then
                    # Make lang2lid
                    # NOTE(kan-bayashi): 0 is reserved for unknown languages
                    echo "<unk> 0" > "${data_feats}${_suf}/${dset}/lang2lid"
                    cut -f 2 -d " " "${data_feats}${_suf}/${dset}/utt2lang" | sort | uniq | \
                        awk '{print $1 " " NR}' >> "${data_feats}${_suf}/${dset}/lang2lid"
                fi
                # NOTE(kan-bayashi): We can reuse the same script for making utt2sid
                pyscripts/utils/utt2spk_to_utt2sid.py \
                    "${data_feats}/org/${train_set}/lang2lid" \
                    "${data_feats}${_suf}/${dset}/utt2lang" \
                    > "${data_feats}${_suf}/${dset}/utt2lid"
            done
        fi
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            if [ -e "${data_feats}/org/${dset}/utt2sid" ]; then
                cp "${data_feats}/org/${dset}/utt2sid" "${data_feats}/${dset}/utt2sid"
            fi
            if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                cp "${data_feats}/org/${dset}/utt2lid" "${data_feats}/${dset}/utt2lid"
            fi

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

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            _utt_extra_files=""
            if [ -e "${data_feats}/org/${dset}/utt2sid" ]; then
                _utt_extra_files+="utt2sid "
            fi
            if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                _utt_extra_files+="utt2lid "
            fi
            # shellcheck disable=SC2086
            utils/fix_data_dir.sh --utt_extra_files "${_utt_extra_files}" "${data_feats}/${dset}"

            # Filter speaker embedding
            # Filter spk_embedding
            if "${use_spk_embed}"; then
                cp "${dumpdir}/${spk_embed_tag}/${dset}"/${spk_embed_tag}.{scp,scp.bak}
                <"${dumpdir}/${spk_embed_tag}/${dset}/${spk_embed_tag}.scp.bak" \
                    utils/filter_scp.pl "${data_feats}/${dset}/wav.scp"  \
                    >"${dumpdir}/${spk_embed_tag}/${dset}/${spk_embed_tag}.scp"
            fi
        done
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Generate src_token_list from ${srctexts}"
        # "nlsyms_txt" should be generated by local/data.sh if need

        # The first symbol in src_token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"

        ${python} -m espnet2.bin.tokenize_text \
              --token_type "${src_token_type}" -f 2- \
              --input "${data_feats}/srctexts" --output "${src_token_list}" \
              --non_linguistic_symbols "${nlsyms_txt}" \
              --cleaner "${cleaner}" \
              --g2p "${g2p}" \
              --write_vocabulary true \
              --add_symbol "${blank}:0" \
              --add_symbol "${oov}:1" \
              --add_symbol "${sos_eos}:-1"

        # NOTE(Jinchuan): also build the tgt_vocab even though this is quite naive.
        #   We may include more control token here to imporve it.
        mkdir -p "${km_dir}"
        mkdir -p "${km_dir}"
        (for n in `seq 1 ${feature_num_clusters}`; do
            echo "<auido_token_${n}>"
        done) > ${tgt_token_list}
    fi

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Discrete TTS discrete unit extraction"

        s3prl_conf="{upstream=${s3prl_upstream_name}}"
        kmeans_feature_type=s3prl
        kmeans_feature_conf="{type=${kmeans_feature_type},conf={s3prl_conf=${s3prl_conf},download_dir=ckpt,multilayer_feature=False,layer=${feature_layer}}}"

        scripts/feats/perform_kmeans.sh \
            --stage ${discrete_stage} \
            --stop_stage ${discrete_stop_stage} \
            --train_set "${train_set}" \
            --dev_set "${valid_set}" \
            --other_sets "${test_sets}" \
            --datadir "${dumpdir}/raw" \
            --featdir "${feature_dir}" \
            --audio_format "${audio_format}" \
            --feature_type ${kmeans_feature_type} \
            --layer "${feature_layer}" \
            --feature_conf "${kmeans_feature_conf}" \
            --km_dir "${km_dir}" \
            --portion "${clustering_portion}" \
            --nclusters "${feature_num_clusters}" \
            --storage_save_mode ${storage_save_mode} \
            --use_gpu "${use_gpu_feat_extract}" \
            --nj ${nj} \
            --num_threads ${clustering_nj} \
            --cpu_cmd "${train_cmd}" \
            --cuda_cmd "${cuda_cmd}"

            # Copy generated pseudo labels to original dump dir
            for dset in ${train_set} ${valid_set} ${test_sets}; do
                _out_dir=${feature_dir}/${kmeans_feature_type}/layer${feature_layer}/${dset}
                cp ${_out_dir}/pseudo_labels_km${feature_num_clusters}.txt \
                    "${data_feats}/${dset}/text.km.${km_tag}"
            done

            # NOTE(jiatong): use the pseudo label without unique to train the vocoder
            log "Saving training pseudo_labels at ${data_feats}/${train_set}/text.km.${km_tag}"
            log "Saving dev pseudo_labels at ${data_feats}/${valid_set}/text.km.${km_tag}"

    fi
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================



if ! "${skip_train}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 7: Discrete TTS collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts2_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _opts+="--write_collected_feats ${write_collected_feats} "

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        # Add extra configs for additional inputs
        # NOTE(kan-bayashi): We always pass this options but not used in default
        _opts+="--pitch_extract_conf fs=${fs} "
        _opts+="--pitch_extract_conf n_fft=${n_fft} "
        _opts+="--pitch_extract_conf hop_length=${n_shift} "
        _opts+="--pitch_extract_conf f0max=${f0max} "
        _opts+="--pitch_extract_conf f0min=${f0min} "
        _opts+="--energy_extract_conf fs=${fs} "
        _opts+="--energy_extract_conf n_fft=${n_fft} "
        _opts+="--energy_extract_conf hop_length=${n_shift} "
        _opts+="--energy_extract_conf win_length=${win_length} "

        if [ -n "${teacher_dumpdir}" ]; then
            _teacher_train_dir="${teacher_dumpdir}/${train_set}"
            _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/durations,durations,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/durations,durations,text_int "
        fi

        if "${use_spk_embed}"; then
            _spk_embed_train_dir="${dumpdir}/${spk_embed_tag}/${train_set}"
            _spk_embed_valid_dir="${dumpdir}/${spk_embed_tag}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_spk_embed_train_dir}/${spk_embed_tag}.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_spk_embed_valid_dir}/${spk_embed_tag}.scp,spembs,kaldi_ark "
        fi

        if "${use_sid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2sid,sids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2sid,sids,text_int "
        fi

        if "${use_lid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2lid,lids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2lid,lids,text_int "
        fi

        # 1. Split the key file
        _logdir="${tts2_stats_dir}/logdir"
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
        log "Generate '${tts2_stats_dir}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${tts2_stats_dir}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${tts2_stats_dir}/run.sh"; chmod +x "${tts2_stats_dir}/run.sh"

        # 3. Submit jobs
        log "Discrete TTS collect_stats started... log: '${_logdir}/stats.*.log'"
        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m "espnet2.bin.${tts2_task}_train" \
                --collect_stats true \
                --write_collected_feats "${write_collected_feats}" \
                --use_preprocessor true \
                --src_token_type "${src_token_type}" \
                --src_token_list "${src_token_list}" \
                --tgt_token_list "${tgt_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --pitch_normalize none \
                --energy_normalize none \
                --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
                --train_data_path_and_name_and_type "${_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_train_dir}/text.km.${km_tag},discrete_speech,text_int" \
                --valid_data_path_and_name_and_type "${_valid_dir}/text,text,text" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_valid_dir}/text.km.${km_tag},discrete_speech,text_int" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${train_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${tts2_stats_dir}"

        # Append the num-src_tokens at the last dimensions. This is used for batch-bins count
        <"${tts2_stats_dir}/train/text_shape" \
            awk -v N="$(<${src_token_list} wc -l)" '{ print $0 "," N }' \
            >"${tts2_stats_dir}/train/text_shape.${src_token_type}"

        <"${tts2_stats_dir}/valid/text_shape" \
            awk -v N="$(<${src_token_list} wc -l)" '{ print $0 "," N }' \
            >"${tts2_stats_dir}/valid/text_shape.${src_token_type}"
    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 8: Discrete TTS Training: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts2_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        if [ -z "${teacher_dumpdir}" ]; then
            #####################################
            #     CASE 1: AR model training     #
            #####################################
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi

            if [ "${num_splits}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${tts2_stats_dir}/splits${num_splits}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    ${python} -m espnet2.bin.split_scps \
                      --scps \
                          "${_train_dir}/text" \
                          "${_train_dir}/${_scp}" \
                          "${tts2_stats_dir}/train/speech_shape" \
                          "${tts2_stats_dir}/train/text_shape.${src_token_type}" \
                      --num_splits "${num_splits}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.km.${km_tag},discrete_speech,text_int "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${src_token_type} "
                _opts+="--train_shape_file ${_split_dir}/discrete_speech_shape "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},speech,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/text.km.${km_tag},discrete_speech,text_int "
                _opts+="--train_shape_file ${tts2_stats_dir}/train/text_shape.${src_token_type} "
                _opts+="--train_shape_file ${tts2_stats_dir}/train/discrete_speech_shape "
            fi
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},speech,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text.km.${km_tag},discrete_speech,text_int "
            _opts+="--valid_shape_file ${tts2_stats_dir}/valid/text_shape.${src_token_type} "
            _opts+="--valid_shape_file ${tts2_stats_dir}/valid/discrete_speech_shape "
        else
            #####################################
            #   CASE 2: Non-AR model training   #
            #####################################
            _teacher_train_dir="${teacher_dumpdir}/${train_set}"
            _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
            _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/durations,durations,text_int "
            _opts+="--train_shape_file ${tts2_stats_dir}/train/text_shape.${src_token_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
            _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/durations,durations,text_int "
            _opts+="--valid_shape_file ${tts2_stats_dir}/valid/text_shape.${src_token_type} "

            # Teacher forcing case: use groundtruth as the target
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/text.km.${km_tag},discrete_speech,text_int "
            _opts+="--train_shape_file ${tts2_stats_dir}/train/speech_shape "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},speech,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text.km.${km_tag},discrete_speech,text_int "
            _opts+="--valid_shape_file ${tts2_stats_dir}/valid/speech_shape "
        fi

        # If there are dumped files of additional inputs, we use it to reduce computational cost
        # NOTE (kan-bayashi): Use dumped files of the target features as well?
        if [ -e "${tts2_stats_dir}/train/collect_feats/pitch.scp" ]; then
            _scp=pitch.scp
            _type=npy
            _train_collect_dir=${tts2_stats_dir}/train/collect_feats
            _valid_collect_dir=${tts2_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},pitch,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},pitch,${_type} "
        fi
        if [ -e "${tts2_stats_dir}/train/collect_feats/energy.scp" ]; then
            _scp=energy.scp
            _type=npy
            _train_collect_dir=${tts2_stats_dir}/train/collect_feats
            _valid_collect_dir=${tts2_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},energy,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},energy,${_type} "
        fi

        # Check extra statistics
        if [ -e "${tts2_stats_dir}/train/pitch_stats.npz" ]; then
            _opts+="--pitch_extract_conf fs=${fs} "
            _opts+="--pitch_extract_conf n_fft=${n_fft} "
            _opts+="--pitch_extract_conf hop_length=${n_shift} "
            _opts+="--pitch_extract_conf f0max=${f0max} "
            _opts+="--pitch_extract_conf f0min=${f0min} "
            _opts+="--pitch_normalize_conf stats_file=${tts2_stats_dir}/train/pitch_stats.npz "
        fi
        if [ -e "${tts2_stats_dir}/train/energy_stats.npz" ]; then
            _opts+="--energy_extract_conf fs=${fs} "
            _opts+="--energy_extract_conf n_fft=${n_fft} "
            _opts+="--energy_extract_conf hop_length=${n_shift} "
            _opts+="--energy_extract_conf win_length=${win_length} "
            _opts+="--energy_normalize_conf stats_file=${tts2_stats_dir}/train/energy_stats.npz "
        fi

        # Add speaker embedding to the inputs if needed
        if "${use_spk_embed}"; then
            _spk_embed_train_dir="${dumpdir}/${spk_embed_tag}/${train_set}"
            _spk_embed_valid_dir="${dumpdir}/${spk_embed_tag}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_spk_embed_train_dir}/${spk_embed_tag}.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_spk_embed_valid_dir}/${spk_embed_tag}.scp,spembs,kaldi_ark "
        fi

        # Add spekaer ID to the inputs if needed
        if "${use_sid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2sid,sids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2sid,sids,text_int "
        fi

        # Add language ID to the inputs if needed
        if "${use_lid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2lid,lids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2lid,lids,text_int "
        fi

        log "Generate '${tts2_exp}/run.sh'. You can resume the process from stage 8 using this script"
        mkdir -p "${tts2_exp}"; echo "${run_args} --stage 8 \"\$@\"; exit \$?" > "${tts2_exp}/run.sh"; chmod +x "${tts2_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "Discrete TTS training started... log: '${tts2_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${tts2_exp})"
        else
            jobname="${tts2_exp}/train.log"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${tts2_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${tts2_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m "espnet2.bin.${tts2_task}_train" \
                --use_preprocessor true \
                --src_token_type "${src_token_type}" \
                --src_token_list "${src_token_list}" \
                --tgt_token_list "${tgt_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --resume true \
                --output_dir "${tts2_exp}" \
                ${_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    tts2_exp="${expdir}/${download_model}"
    mkdir -p "${tts2_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${tts2_exp}/config.txt"

    # Get the path of each file
    _model_file=$(<"${tts2_exp}/config.txt" sed -e "s/.*'model_file': '\([^']*\)'.*$/\1/")
    _train_config=$(<"${tts2_exp}/config.txt" sed -e "s/.*'train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_model_file}" "${tts2_exp}"
    ln -sf "${_train_config}" "${tts2_exp}"
    inference_model=$(basename "${_model_file}")

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Decoding: training_dir=${tts2_exp}"

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

        log "Generate '${tts2_exp}/${inference_tag}/run.sh'. You can resume the process from stage 9 using this script"
        mkdir -p "${tts2_exp}/${inference_tag}"; echo "${run_args} --stage 9 \"\$@\"; exit \$?" > "${tts2_exp}/${inference_tag}/run.sh"; chmod +x "${tts2_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _speech_data="${_data}"
            _dir="${tts2_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/log"
            mkdir -p "${_logdir}"

            _ex_opts=""
            if [ -n "${teacher_dumpdir}" ]; then
                # Use groundtruth of durations
                _teacher_dir="${teacher_dumpdir}/${dset}"
                _ex_opts+="--data_path_and_name_and_type ${_teacher_dir}/durations,durations,text_int "
                # Overwrite speech arguments if use knowledge distillation
                if [ -e "${teacher_dumpdir}/${train_set}/probs" ]; then
                    _speech_data="${_teacher_dir}/denorm"
                    _scp=feats.scp
                    _type=npy
                fi
            fi

            # Add speaker embedding to the inputs if needed
            if "${use_spk_embed}"; then
                _spk_embed_dir="${dumpdir}/${spk_embed_tag}/${dset}"
                _ex_opts+="--data_path_and_name_and_type ${_spk_embed_dir}/${spk_embed_tag}.scp,spembs,kaldi_ark "
            fi

            # Add spekaer ID to the inputs if needed
            if "${use_sid}"; then
                _ex_opts+="--data_path_and_name_and_type ${_data}/utt2sid,sids,text_int "
            fi

            # Add language ID to the inputs if needed
            if "${use_lid}"; then
                _ex_opts+="--data_path_and_name_and_type ${_data}/utt2lid,lids,text_int "
            fi

            # 0. Copy feats_type
            cp "${_data}/feats_type" "${_dir}/feats_type"

            # 1. Split the key file
            key_file=${_data}/text
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/tts2_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/tts2_inference.JOB.log \
                ${python} -m espnet2.bin.tts2_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/text,text,text" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --model_file "${tts2_exp}"/"${inference_model}" \
                    --train_config "${tts2_exp}"/config.yaml \
                    --output_dir "${_logdir}"/output.JOB \
                    --vocoder_file "${vocoder_file}" \
                    ${_opts} ${_ex_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/tts2_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            if [ -e "${_logdir}/output.${_nj}/feats" ]; then
                mkdir -p "${_dir}"/feats
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/feats/feats.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/feats/feats.scp"
            fi
            if [ -e "${_logdir}/output.${_nj}/speech_shape" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/speech_shape/speech_shape"
                done | LC_ALL=C sort -k1 > "${_dir}/speech_shape"
            fi
            if [ -e "${_logdir}/output.${_nj}/wav" ]; then
                mkdir -p "${_dir}"/wav
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
                    rm -rf "${_logdir}/output.${i}"/wav
                done
                find "${_dir}/wav" -name "*.wav" | while read -r line; do
                    echo "$(basename "${line}" .wav) ${line}"
                done | LC_ALL=C sort -k1 > "${_dir}/wav/wav.scp"
            fi
            if [ -e "${_logdir}/output.${_nj}/att_ws" ]; then
                mkdir -p "${_dir}"/att_ws
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/att_ws/*.png "${_dir}"/att_ws
                    rm -rf "${_logdir}/output.${i}"/att_ws
                done
            fi
            if [ -e "${_logdir}/output.${_nj}/durations" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/durations/durations"
                done | LC_ALL=C sort -k1 > "${_dir}/durations"
            fi
            if [ -e "${_logdir}/output.${_nj}/focus_rates" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/focus_rates/focus_rates"
                done | LC_ALL=C sort -k1 > "${_dir}/focus_rates"
            fi
            if [ -e "${_logdir}/output.${_nj}/probs" ]; then
                mkdir -p "${_dir}"/probs
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/probs/*.png "${_dir}"/probs
                    rm -rf "${_logdir}/output.${i}"/probs
                done
            fi
        done
    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${tts2_exp}/${tts2_exp##*/}_${inference_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Stage 10: Pack model: ${packed_model}"
        log "Warning: Upload model to Zenodo will be deprecated. We encourage to use Hugging Face"

        _opts=""
        if "${use_spk_embed}"; then
            for dset in "${train_set}" ${test_sets}; do
                _opts+=" --option ${dumpdir}/${spk_embed_tag}/${dset}/spk_${spk_embed_tag}.scp"
                _opts+=" --option ${dumpdir}/${spk_embed_tag}/${dset}/spk_${spk_embed_tag}.ark"
            done
        fi
        if "${use_sid}"; then
            _opts+=" --option ${data_feats}/org/${train_set}/spk2sid"
        fi
        if "${use_lid}"; then
            _opts+=" --option ${data_feats}/org/${train_set}/lang2lid"
        fi

	_km_dir="${km_dir}/${s3prl_upstream_name}_layer6${feature_layer}_${feature_num_clusters}"
        _opts+=" --option ${_km_dir}/km_${feature_num_clusters}.mdl "
        ${python} -m espnet2.bin.pack tts2 \
            --train_config "${tts2_exp}"/config.yaml \
            --model_file "${tts2_exp}"/"${inference_model}" \
            --option "${tts2_exp}"/images  \
            --outpath "${packed_model}" \
            ${_opts}

        # NOTE(kamo): If you'll use packed model to inference in this script, do as follows
        #   % unzip ${packed_model}
        #   % ./run.sh --stage 8 --tts2_exp $(basename ${packed_model} .zip) --inference_model pretrain.pth
    fi
fi

if ! "${skip_upload}"; then
    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        log "Stage 11: Upload model to Zenodo: ${packed_model}"

        # To upload your model, you need to do:
        #   1. Signup to Zenodo: https://zenodo.org/
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
        # /some/where/espnet/egs2/foo/tts2/ -> foo/tts2
        _task="$(pwd | rev | cut -d/ -f1-2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${tts2_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
# Download the model file here
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Config</strong><pre><code>$(cat "${tts2_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${tts2_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stage"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace" && \
            exit 1
        log "Stage 12: Upload model to HuggingFace: ${hf_repo}"

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
        # /some/where/espnet/egs2/foo/asr1/ -> foo/asr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # copy files in ${dir_repo}
        unzip -o ${packed_model} -d ${dir_repo}
        # Generate description file
        # shellcheck disable=SC2034
        hf_task=text-to-speech
        # shellcheck disable=SC2034
        espnet_task=TTS2
        # shellcheck disable=SC2034
        task_exp=${tts2_exp}
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
else
    log "Skip the uploading to HuggingFace stage"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
