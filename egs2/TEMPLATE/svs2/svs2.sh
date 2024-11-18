#!/usr/bin/env bash

# Credit to espnet asr.sh

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
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
gpu_id=0             # GPU_id, only works when ngpu=1
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet2 commands.

# Data preparation related
local_data_opts="" # Options to be passed to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (fbank or stft or raw).
audio_format=wav     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.
use_sid=false        # Whether to use speaker id as the inputs (Need utt2spk in data directory).
use_lid=false        # Whether to use language id as the inputs (Need utt2lang in data directory).
use_xvector=false    # Whether to use x-vector
feats_extract=fbank        # On-the-fly feature extractor.
feats_normalize=global_mvn # On-the-fly feature normalizer.
# Only used for feats_type != raw
fs=16000          # Sampling rate.
fmin=80           # Minimum frequency of Mel basis.
fmax=12000        # Maximum frequency of Mel basis.
n_mels=80         # The number of mel basis.
n_fft=1024        # The number of fft points.
n_shift=256       # The number of shift points.
win_length=null   # Window length.
score_feats_extract=frame_score_feats # The type of music score feats (frame_score_feats or syllable_score_feats)
pitch_extract=None
ying_extract=None
# Only used for the model using pitch features (e.g. FastSpeech2)
f0min=80          # Maximum f0 for pitch extraction.
f0max=800         # Minimum f0 for pitch extraction.

oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol.
sos_eos="<sos/eos>" # sos and eos symbols.

# Kmeans related
kmeans_opts=                # The options given to scripts/feats/perform_kmeans.sh
kmeans_feature="wavlm_large/21" # format: ssl_model_type/layer_idx (e.g. mfcc, hubert_large/21, wavlm_large/21)
multi_token=None
RVQ_layers=1
token_name=""
discrete_token_layers=1
mix_type="frame"
portion=1.0
nclusters=2000              # The number of clusters for discrete tokenss
storage_save_mode=true      # Save storage on SSL feature extraction
                            # If true, feature extraction and kmeans clustering on the fly

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
token_joint=false       # whether to use a single bpe system for both source and target languages

# Training related
train_config=""    # Config for training.
train_args=""      # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
svs_exp=""         # Specify the direcotry path for experiment. If this option is specified, tag is ignored.
svs_stats_dir=""   # Specify the direcotry path for statistics. If empty, automatically decided.
num_splits=1       # Number of splitting for svs corpus.
teacher_dumpdir="" # Directory of teacher outpus
write_collected_feats=false # Whether to dump features in stats collection.
svs_task=svs                # SVS task (svs or gan_svs)
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch

# Decoding related
inference_config="" # Config for decoding.
inference_args=""   # Arguments for decoding, e.g., "--threshold 0.75".
                    # Note that it will overwrite args in inference config.
inference_tag=""    # Suffix for decoding directory.
inference_model=valid.loss.best.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
vocoder_file=none  # Vocoder parameter file, If set to none, Griffin-Lim will be used.
download_model=""   # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""     # Name of training set.
valid_set=""     # Name of validation set used for monitoring/tuning network training.
test_sets=""     # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
srctexts=""      # Texts to create token list. Multiple items can be specified.
nlsyms_txt=none  # Non-linguistic symbol list (needed if existing).
token_type=phn   # Transcription type.
cleaner=none     # Text cleaner.
g2p=g2p_en       # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
text_fold_length=150   # fold_length for text data.
singing_fold_length=800 # fold_length for singing data.

nlsyms_txt=none  # Non-linguistic symbol list if existing.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.

# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

preset_layer=
preset_token=

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
    --python         # Specify python to execute espnet2 commands (default="${python}").

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (fbank or stft or raw, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --use_sid          # Whether to use speaker id as the inputs (default="${use_sid}").
    --use_lid          # Whether to use language id as the inputs (default="${use_lid}").
    --use_xvector      # Whether to use X-vector (Require Kaldi, default="${use_xvector}").
    --fs               # Sampling rate (default="${fs}").
    --fmax             # Maximum frequency of Mel basis (default="${fmax}").
    --fmin             # Minimum frequency of Mel basis (default="${fmin}").
    --n_mels           # The number of mel basis (default="${n_mels}").
    --n_fft            # The number of fft points (default="${n_fft}").
    --n_shift          # The number of shift points (default="${n_shift}").
    --win_length       # Window length (default="${win_length}").
    --f0min            # Maximum f0 for pitch extraction (default="${f0min}").
    --f0max            # Minimum f0 for pitch extraction (default="${f0max}").
    --oov              # Out of vocabrary symbol (default="${oov}").
    --blank            # CTC blank symbol (default="${blank}").
    --sos_eos          # sos and eos symbole (default="${sos_eos}").

    # Training related
    --train_config  # Config for training (default="${train_config}").
    --train_args    # Arguments for training (default="${train_args}").
                    # e.g., --train_args "--max_epoch 1"
                    # Note that it will overwrite args in train config.
    --tag           # Suffix for training directory (default="${tag}").
    --svs_exp       # Specify the direcotry path for experiment.
                    # If this option is specified, tag is ignored (default="${svs_exp}").
    --svs_stats_dir # Specify the direcotry path for statistics.
                    # If empty, automatically decided (default="${svs_stats_dir}").
    --num_splits    # Number of splitting for svs corpus (default="${num_splits}").
    --teacher_dumpdir       # Direcotry of teacher outputs
    --write_collected_feats # Whether to dump features in statistics collection (default="${write_collected_feats}").
    --svs_task              # SVS task (svs or gan_svs, now only support svs)
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").

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
    --srctexts           # Texts to create token list (required).
                         # Note that multiple items can be specified.
    --nlsyms_txt         # Non-linguistic symbol list (default="${nlsyms_txt}").
    --token_type         # Transcription type (default="${token_type}").
    --cleaner            # Text cleaner (default="${cleaner}").
    --g2p                # g2p method (default="${g2p}").
    --lang               # The language type of corpus (default="${lang}").
    --text_fold_length   # Fold length for text data (default="${text_fold_length}").
    --singing_fold_length # Fold length for singing data (default="${singing_fold_length}").
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

. ./path.sh || exit 1
. ./cmd.sh

# TODO(Yuning): Check required arguments

# Check feature type
if [ "${feats_type}" = fbank ]; then
    data_feats="${dumpdir}/fbank"
elif [ "${feats_type}" = stft ]; then
    data_feats="${dumpdir}/stft"
elif [ "${feats_type}" = raw ]; then
    #data_audio="${dumpdir}/audio_raw"
    data_feats="${dumpdir}/${feats_type}"
    data_extract="${dumpdir}/extracted"
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Extra files for SVS
utt_extra_files="label score.scp"
train_sp_sets=
# Check token list type
token_listdir="data/token_list/${token_type}"
if [ "${cleaner}" != none ]; then
    token_listdir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    token_listdir+="_${g2p}"
fi
if [ "${lang}" != noinfo ]; then
    token_listdir+="_${lang}"
fi
token_list="${token_listdir}/tokens.txt"

mert_url="m-a-p/MERT-v1-330M"
encodec_url="facebook/encodec_48khz"
if [ ${kmeans_feature} = "mfcc" ]; then  # MFCC has no layer
    kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
    layer=
    kmeans_feature_conf="{type=mfcc}"
else
    kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
    layer=$(echo "${kmeans_feature}" | cut -d/ -f2)
    # TODO(simpleoier): to support features beyond s3prl
    if [ ${kmeans_feature_type} = "mert" ]; then
        kmeans_feature_conf="{type=mert,conf={fs=24000,multilayer_feature=False,layer=${layer},download_path=${mert_url}}}"
    elif [ ${kmeans_feature_type} = "encodec" ]; then
        kmeans_feature_conf="{type=encodec,conf={fs=48000,bandwidth=12,multilayer_feature=False,layer=${layer},download_path=${encodec_url}}}"
    elif [ ${kmeans_feature_type} != "multi" ]; then
        s3prl_conf="{upstream=${kmeans_feature_type}}"
        kmeans_feature_conf="{type=s3prl,conf={s3prl_conf=${s3prl_conf},download_dir=ckpt,multilayer_feature=False,layer=${layer}}}"
    fi
fi
if [ ${kmeans_feature_type} = "multi" ]; then
    token_name=${layer}
    discrete_token_layers=$(echo "$multi_token" | tr -cd ' ' | wc -c)
    ((discrete_token_layers++))
    token_file=token_multi_${token_name}
else
    token_name=${kmeans_feature_type}_${nclusters}_${layer}
    multi_token=${token_name}
    if [ ${kmeans_feature} = "mfcc" ]; then
        token_file=token_${kmeans_feature_type}_${nclusters}
    else
        token_file=token_${kmeans_feature_type}_${nclusters}_${layer}
    fi
    km_dir="${expdir}"/kmeans/$(echo "${kmeans_feature}" | tr "/" "_")_${nclusters}clusters
fi

if [ "$preset_layer" != none  ]; then
    discrete_token_layers=${preset_layer}
fi

if [ "$preset_token" != none ]; then
    echo "hello: $preset_token"
    token_file=${preset_token}
fi

# token_file=token_multi_ad_3rs
# discrete_token_layers=3

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)_${feats_type}_${token_type}"
    else
        tag="train_${feats_type}_${token_type}"
    fi
    if [ "${cleaner}" != none ]; then
        tag+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        tag+="_${g2p}"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi

    if [ "${lang}" != noinfo ]; then
        tag+="_${lang}"
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
if [ -z "${svs_stats_dir}" ]; then
    svs_stats_dir="${expdir}/svs_stats_${feats_type}_${token_type}"
    if [ "${cleaner}" != none ]; then
        svs_stats_dir+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        svs_stats_dir+="_${g2p}"
    fi
    if [ "${lang}" != noinfo ]; then
        svs_stats_dir+="_${lang}"
    fi
fi

# The directory used for training commands
if [ -z "${svs_exp}" ]; then
    svs_exp="${expdir}/svs_${tag}"
fi


# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts} --fs "${fs}" --g2p "${g2p}"
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

        if [ "${feats_type}" = raw ]; then
            log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets} ; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh "data/${dset}" "${data_feats}${_suf}/${dset}"
                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                for extra_file in ${utt_extra_files}; do
                    for single_file in $(ls data/"${dset}"/${extra_file}); do
                        cp ${single_file} "${data_feats}${_suf}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}" "${data_feats}${_suf}/${dset}"
                for extra_file in ${expand_utt_extra_files}; do
                    LC_ALL=C sort -u -k1,1 "${data_feats}${_suf}/${dset}/${extra_file}" -o "${data_feats}${_suf}/${dset}/${extra_file}"
                done
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    _opts+="--segments data/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
                scripts/audio/format_score_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    ${_opts} \
                    "data/${dset}/score.scp" "${data_feats}${_suf}/${dset}"
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done
        fi
    fi
    # TODO(Yuning): Introducing a single new stage for conditional token generation
    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if "${use_sid}"; then
            log "Stage 2+: sid extract: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1.Generate spk2sid

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

		utt_extra_files="${utt_extra_files} utt2sid"
            done
        fi
    fi


    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if "${use_lid}"; then
            log "Stage 2+: lid extract: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1.Generate spk2sid

                if [ "${dset}" = "${train_set}" ]; then
                    # Make spk2sid
                    # NOTE(kan-bayashi): 0 is reserved for unknown speakers
                    echo "<unk> 0" > "${data_feats}${_suf}/${dset}/lang2lid"
                    cut -f 2 -d " " "${data_feats}${_suf}/${dset}/utt2lang" | sort | uniq | \
                        awk '{print $1 " " NR}' >> "${data_feats}${_suf}/${dset}/lang2lid"
                fi
                pyscripts/utils/utt2spk_to_utt2sid.py \
                    "${data_feats}/org/${train_set}/lang2lid" \
                    "${data_feats}${_suf}/${dset}/utt2lang" \
                    > "${data_feats}${_suf}/${dset}/utt2lid"

		utt_extra_files="${utt_extra_files} utt2lid"
            done
        fi
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            for utt_extra_file in ${utt_extra_files}; do
                cp "${data_feats}/org/${dset}/${utt_extra_file}" "${data_feats}/${dset}"
            done
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
            fi

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" "${data_feats}/${dset}"

        done

        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        if [ ${kmeans_feature_type} = "encodec" ]; then
            log "Stage 4a: Extract ${kmeans_feature_type} tokens"

            _cmd="${cuda_cmd} --gpu 1"

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                _dump_dir="${data_extract}/${kmeans_feature_type}/layer${layer}/${dset}"

                utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/${dset}" "${_dump_dir}"

                mkdir -p "${_dump_dir}"
                mkdir -p "${_dump_dir}/data"
                mkdir -p "${_dump_dir}/logdir"
                nutt=$(<"${_dump_dir}"/wav.scp wc -l)
                _nj=$((nj<nutt?nj:nutt))

                key_file="${data_feats}/${dset}"/wav.scp
                split_scps=""

                for n in $(seq ${_nj}); do
                    split_scps+=" ${_dump_dir}/logdir/wav.${n}.scp"
                done
                # shellcheck disable=SC2086
                utils/split_scp.pl "${key_file}" ${split_scps}

                for n in $(seq ${_nj}); do
                    awk '(FILENAME==ARGV[1]){utt2num[$1]=$2} (FILENAME==ARGV[2]){print($1, utt2num[$1])}' \
                        ${data_feats}/${dset}/utt2num_samples ${_dump_dir}/logdir/wav.${n}.scp > ${_dump_dir}/logdir/utt2num_samples.${n}
                done

                ${_cmd} JOB=1:${_nj} ${_dump_dir}/logdir/dump_features.JOB.log \
                    ${python} pyscripts/feats/dump_ssl_feature.py \
                        --feature_conf "'${kmeans_feature_conf}'" \
                        --use_gpu true \
                        --in_filetype "sound" \
                        --out_filetype "mat" \
                        --write_num_frames "ark,t:${_dump_dir}/data/utt2num_frames.JOB" \
                        --utt2num_samples "${_dump_dir}/logdir/utt2num_samples.JOB" \
                        ${batch_bins:+--batch_bins ${batch_bins}} \
                        "scp:${_dump_dir}/logdir/wav.JOB.scp" \
                        "ark,t:${_dump_dir}/logdir/pseudo_labels_km${nclusters}.JOB.txt" || exit 1;

                # concatenate scp files
                for n in $(seq ${_nj}); do
                    cat "${_dump_dir}"/logdir/pseudo_labels_km${nclusters}.${n}.txt || exit 1;
                done | sed 's/ \[ \| \]//g' | sort -u > "${_dump_dir}"/pseudo_labels_km${nclusters}.txt || exit 1;

            done
        elif [ ${kmeans_feature_type} != "multi" ]; then
            log "Stage 4a: Perform Kmeans using ${kmeans_feature_type} features"
            nj=1
            scripts/feats/perform_kmeans.sh \
                --stage 1 --stop-stage 3 \
                --train_set "${train_set}" \
                --dev_set "${valid_set}" \
                --other_sets "${test_sets} ${train_sp_sets}" \
                --datadir "${data_feats}" \
                --featdir "${data_extract}" \
                --audio_format "${audio_format}" \
                --feature_type "${kmeans_feature_type}" \
                --layer "${layer}" \
                --RVQ_layers "${RVQ_layers}" \
                --feature_conf "${kmeans_feature_conf}" \
                --km_dir "${km_dir}" \
                --portion "${portion}" \
                --nclusters "${nclusters}" \
                --storage_save_mode ${storage_save_mode} \
                --use_gpu true \
                --nj ${nj} \
                --cpu_cmd "${train_cmd}" \
                --cuda_cmd "${cuda_cmd}" \
                ${kmeans_opts}
        fi
        if [ ${kmeans_feature_type} != "multi" ]; then
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ ${kmeans_feature_type} = "mfcc" ]; then
                    _dump_dir="${data_extract}/${kmeans_feature_type}/${dset}"
                else
                    _dump_dir="${data_extract}/${kmeans_feature_type}/layer${layer}/${dset}"
                fi
                token_files=""
                if [ ${RVQ_layers} = 1 ]; then
                    cp "${_dump_dir}/pseudo_labels_km${nclusters}.txt" "${data_feats}/${dset}/${token_file}"
                    token_files="${token_file}"
                else
                    for i in $(seq ${RVQ_layers}); do
                        cp "${_dump_dir}/pseudo_labels_RVQ_$((i-1))_km${nclusters}.txt" "${data_feats}/${dset}/${token_file}_RVQ_$((i-1))"
                        token_files+="${token_file}_RVQ_$((i-1)) "
                    done
                fi
                # fix_data_dir.sh leaves only utts which exist in all files
                utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files} ${token_file}" "${data_feats}/${dset}"
            done
        else
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                pyscripts/feats/combine_tokens.py \
                    --dir "${data_feats}/${dset}/" \
                    --tokens "${multi_token}" \
                    --target ${token_file} \
                    --mix_type ${mix_type}
                utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files} token_multi_${token_name}" "${data_feats}/${dset}"
            done
        fi
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5a: Generate token_list from ${srctexts}"
        # "nlsyms_txt" should be generated by local/data.sh if need

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

        ${python} -m espnet2.bin.tokenize_text \
              --token_type "${token_type}" -f 2- \
              --input "${data_feats}/srctexts" --output "${token_list}" \
              --non_linguistic_symbols "${nlsyms_txt}" \
              --cleaner "${cleaner}" \
              --g2p "${g2p}" \
              --write_vocabulary true \
              --add_symbol "${blank}:0" \
              --add_symbol "${oov}:1" \
              --add_symbol "${sos_eos}:-1"
    fi
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================

if ! "${skip_train}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: SVS collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.svs_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi
        _opts+="--feats_extract ${feats_extract} "
        _opts+="--feats_extract_conf n_fft=${n_fft} "
        _opts+="--feats_extract_conf hop_length=${n_shift} "
        _opts+="--feats_extract_conf win_length=${win_length} "
        if [ "${feats_extract}" = fbank ]; then
            _opts+="--feats_extract_conf fs=${fs} "
            _opts+="--feats_extract_conf fmin=${fmin} "
            _opts+="--feats_extract_conf fmax=${fmax} "
            _opts+="--feats_extract_conf n_mels=${n_mels} "
        fi

        # Add extra configs for additional inputs
        # NOTE(kan-bayashi): We always pass this options but not used in default
        _opts+="--score_feats_extract ${score_feats_extract} "
        _opts+="--score_feats_extract_conf fs=${fs} "
        _opts+="--score_feats_extract_conf n_fft=${n_fft} "
        _opts+="--score_feats_extract_conf win_length=${win_length} "
        _opts+="--score_feats_extract_conf hop_length=${n_shift} "
        _opts+="--pitch_extract ${pitch_extract} "
        _opts+="--pitch_extract_conf fs=${fs} "
        _opts+="--pitch_extract_conf n_fft=${n_fft} "
        _opts+="--pitch_extract_conf hop_length=${n_shift} "
        _opts+="--pitch_extract_conf f0max=${f0max} "
        _opts+="--pitch_extract_conf f0min=${f0min} "
        _opts+="--ying_extract ${ying_extract} "
        _opts+="--ying_extract_conf fs=${fs} "
        _opts+="--ying_extract_conf w_step=${n_shift} "
        _opts+="--energy_extract_conf fs=${fs} "
        _opts+="--energy_extract_conf n_fft=${n_fft} "
        _opts+="--energy_extract_conf hop_length=${n_shift} "
        _opts+="--energy_extract_conf win_length=${win_length} "
        _opts+="--discrete_token_layers ${discrete_token_layers} "
        _opts+="--nclusters ${nclusters} "

        if [ -n "${teacher_dumpdir}" ]; then
            _teacher_train_dir="${teacher_dumpdir}/${train_set}"
            _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/label,label,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/label,label,text_int "
        fi

        if "${use_xvector}"; then
            _xvector_train_dir="${dumpdir}/xvector/${train_set}"
            _xvector_valid_dir="${dumpdir}/xvector/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
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
        _logdir="${svs_stats_dir}/logdir"
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
        log "Generate '${svs_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${svs_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${svs_stats_dir}/run.sh"; chmod +x "${svs_stats_dir}/run.sh"

        # 3. Submit jobs
        log "SVS collect_stats started... log: '${_logdir}/stats.*.log'"
        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m "espnet2.bin.${svs_task}_train" \
                --collect_stats true \
                --write_collected_feats "${write_collected_feats}" \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize none \
                --pitch_normalize none \
                --energy_normalize none \
                --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
                --train_data_path_and_name_and_type "${_train_dir}/label,label,duration" \
                --train_data_path_and_name_and_type "${_train_dir}/score.scp,score,score" \
                --train_data_path_and_name_and_type "${_train_dir}/${_scp},singing,${_type}" \
                --train_data_path_and_name_and_type "${_train_dir}/${token_file},discrete_token,text_int" \
                --valid_data_path_and_name_and_type "${_valid_dir}/text,text,text" \
                --valid_data_path_and_name_and_type "${_valid_dir}/label,label,duration" \
                --valid_data_path_and_name_and_type "${_valid_dir}/score.scp,score,score" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${_scp},singing,${_type}" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${token_file},discrete_token,text_int" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                --fs ${fs} \
                ${_opts} ${train_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done

        if [ "${feats_normalize}" != global_mvn ]; then
            # Skip summerizaing stats if not using global MVN
            _opts+="--skip_sum_stats"
        fi
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${svs_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${svs_stats_dir}/train/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${svs_stats_dir}/train/text_shape.${token_type}"

        <"${svs_stats_dir}/valid/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${svs_stats_dir}/valid/text_shape.${token_type}"
    fi



    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 7: SVS Training: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.svs_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        if [ -z "${teacher_dumpdir}" ]; then
            log "CASE 1: AR model training or NAR model with music score duration"
            ############################################################################
            #     CASE 1: AR model training or NAR model with music score duration     #
            ############################################################################
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            _type=sound
            _fold_length="$((singing_fold_length * n_shift))"
            _opts+="--score_feats_extract ${score_feats_extract} "
            _opts+="--score_feats_extract_conf fs=${fs} "
            _opts+="--score_feats_extract_conf n_fft=${n_fft} "
            _opts+="--score_feats_extract_conf win_length=${win_length} "
            _opts+="--score_feats_extract_conf hop_length=${n_shift} "
            _opts+="--feats_extract ${feats_extract} "
            _opts+="--feats_extract_conf n_fft=${n_fft} "
            _opts+="--feats_extract_conf hop_length=${n_shift} "
            _opts+="--feats_extract_conf win_length=${win_length} "
            _opts+="--pitch_extract ${pitch_extract} "
            _opts+="--ying_extract ${ying_extract} "
            if [ "${feats_extract}" = fbank ]; then
                _opts+="--feats_extract_conf fs=${fs} "
                _opts+="--feats_extract_conf fmin=${fmin} "
                _opts+="--feats_extract_conf fmax=${fmax} "
                _opts+="--feats_extract_conf n_mels=${n_mels} "
            fi
            if [ "${pitch_extract}" = dio ]; then
                _opts+="--pitch_extract_conf fs=${fs} "
                _opts+="--pitch_extract_conf n_fft=${n_fft} "
                _opts+="--pitch_extract_conf hop_length=${n_shift} "
                _opts+="--pitch_extract_conf f0max=${f0max} "
                _opts+="--pitch_extract_conf f0min=${f0min} "
            fi
            if [ "${ying_extract}" = ying ]; then
                _opts+="--ying_extract_conf fs=${fs} "
                _opts+="--ying_extract_conf w_step=${n_shift} "
            fi

            if [ "${num_splits}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${svs_stats_dir}/splits${num_splits}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    ${python} -m espnet2.bin.split_scps \
                      --scps \
                          "${_train_dir}/text" \
                          "${_train_dir}/${_scp}" \
                          "${svs_stats_dir}/train/singing_shape" \
                          "${svs_stats_dir}/train/text_shape.${token_type}" \
                      --num_splits "${num_splits}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},singing,${_type} "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
                _opts+="--train_shape_file ${_split_dir}/singing_shape "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},singing,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/label,label,duration "
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/score.scp,score,score "
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/${token_file},discrete_token,text_int "
                # echo "svs_stats_dir: ${svs_stats_dir}"

                _opts+="--train_shape_file ${svs_stats_dir}/train/text_shape.${token_type} "
                _opts+="--train_shape_file ${svs_stats_dir}/train/singing_shape "
            fi
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},singing,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/label,label,duration "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/score.scp,score,score "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${token_file},discrete_token,text_int "
            _opts+="--valid_shape_file ${svs_stats_dir}/valid/text_shape.${token_type} "
            _opts+="--valid_shape_file ${svs_stats_dir}/valid/singing_shape "
        else
	    log "CASE 2: Non-AR model training (with additional alignment)"
            ##################################################################
	    #   CASE 2: Non-AR model training  (with additional alignment)   #
            ##################################################################
            _teacher_train_dir="${teacher_dumpdir}/${train_set}"
            _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
            _fold_length="${singing_fold_length}"
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
            _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/label,label,text_int "
            _opts+="--train_shape_file ${svs_stats_dir}/train/text_shape.${token_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
            _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/label,label,text_int "
            _opts+="--valid_shape_file ${svs_stats_dir}/valid/text_shape.${token_type} "

            if [ -e ${_teacher_train_dir}/probs ]; then
                # Knowledge distillation case: use the outputs of the teacher model as the target
                _scp=feats.scp
                _type=npy
                _odim="$(head -n 1 "${_teacher_train_dir}/singing_shape" | cut -f 2 -d ",")"
                _opts+="--odim=${_odim} "
                _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/denorm/${_scp},singing,${_type} "
                _opts+="--train_shape_file ${_teacher_train_dir}/singing_shape "
                _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/denorm/${_scp},singing,${_type} "
                _opts+="--valid_shape_file ${_teacher_valid_dir}/singing_shape "
            else
                # Teacher forcing case: use groundtruth as the target
                _scp=wav.scp
                _type=sound
                _fold_length="$((singing_fold_length * n_shift))"
                _opts+="--feats_extract ${feats_extract} "
                _opts+="--feats_extract_conf n_fft=${n_fft} "
                _opts+="--feats_extract_conf hop_length=${n_shift} "
                _opts+="--feats_extract_conf win_length=${win_length} "
                if [ "${feats_extract}" = fbank ]; then
                    _opts+="--feats_extract_conf fs=${fs} "
                    _opts+="--feats_extract_conf fmin=${fmin} "
                    _opts+="--feats_extract_conf fmax=${fmax} "
                    _opts+="--feats_extract_conf n_mels=${n_mels} "
                fi
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},singing,${_type} "
                _opts+="--train_shape_file ${svs_stats_dir}/train/singing_shape "
                _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},singing,${_type} "
                _opts+="--valid_shape_file ${svs_stats_dir}/valid/singing_shape "
            fi
        fi

        # NOTE (jiatong): Use dumped files of the target features as well
        if [ -e "${svs_stats_dir}/train/collect_feats/pitch.scp" ]; then
            _scp=pitch.scp
            _type=npy
            _train_collect_dir=${svs_stats_dir}/train/collect_feats
            _valid_collect_dir=${svs_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},pitch,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},pitch,${_type} "
        fi
        if [ -e "${svs_stats_dir}/train/collect_feats/energy.scp" ]; then
            _scp=energy.scp
            _type=npy
            _train_collect_dir=${svs_stats_dir}/train/collect_feats
            _valid_collect_dir=${svs_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},energy,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},energy,${_type} "
        fi
        if [ -e "${svs_stats_dir}/train/collect_feats/feats.scp" ]; then
            _scp=feats.scp
            _type=npy
            _train_collect_dir=${svs_stats_dir}/train/collect_feats
            _valid_collect_dir=${svs_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},feats,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},feats,${_type} "
        fi
        if [ -e "${svs_stats_dir}/train/collect_feats/ying.scp" ]; then
            _scp=ying.scp
            _type=npy
            _train_collect_dir=${svs_stats_dir}/train/collect_feats
            _valid_collect_dir=${svs_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},ying,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},ying,${_type} "
        fi

        # Check extra statistics
        if [ -e "${svs_stats_dir}/train/pitch_stats.npz" ]; then
            _opts+="--pitch_extract_conf fs=${fs} "
            _opts+="--pitch_extract_conf n_fft=${n_fft} "
            _opts+="--pitch_extract_conf hop_length=${n_shift} "
            _opts+="--pitch_extract_conf f0max=${f0max} "
            _opts+="--pitch_extract_conf f0min=${f0min} "
            _opts+="--pitch_normalize_conf stats_file=${svs_stats_dir}/train/pitch_stats.npz "
        fi
        if [ -e "${svs_stats_dir}/train/energy_stats.npz" ]; then
            _opts+="--energy_extract_conf fs=${fs} "
            _opts+="--energy_extract_conf n_fft=${n_fft} "
            _opts+="--energy_extract_conf hop_length=${n_shift} "
            _opts+="--energy_extract_conf win_length=${win_length} "
            _opts+="--energy_normalize_conf stats_file=${svs_stats_dir}/train/energy_stats.npz "
        fi
        if [ -e "${svs_stats_dir}/train/ying_stats.npz" ]; then
            _opts+="--ying_extract_conf fs=${fs} "
            _opts+="--ying_extract_conf w_step=${n_shift} "
        fi


        # Add X-vector to the inputs if needed
        if "${use_xvector}"; then
            _xvector_train_dir="${dumpdir}/xvector/${train_set}"
            _xvector_valid_dir="${dumpdir}/xvector/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
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

        if [ "${feats_normalize}" = "global_mvn" ]; then
            _opts+="--normalize_conf stats_file=${svs_stats_dir}/train/feats_stats.npz "
        fi

        log "Generate '${svs_exp}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${svs_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${svs_exp}/run.sh"; chmod +x "${svs_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "SVS training started... log: '${svs_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${svs_exp})"
        else
            jobname="${svs_exp}/train.log"
        fi

        _opts+="--discrete_token_layers ${discrete_token_layers} "
        _opts+="--nclusters ${nclusters} "

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${svs_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${svs_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m "espnet2.bin.${svs_task}_train" \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --fs "${fs}" \
                --normalize "${feats_normalize}" \
                --resume true \
                --init_param ${pretrained_model} \
                --ignore_init_mismatch ${ignore_init_mismatch} \
                --fold_length "${text_fold_length}" \
                --fold_length "${_fold_length}" \
                --output_dir "${svs_exp}" \
                ${_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi

if ! "${skip_eval}"; then
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Decoding: training_dir=${svs_exp}"

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

        if [ -z "${teacher_dumpdir}" ]; then
            _feats_type="$(<${data_feats}/${train_set}/feats_type)"
        else
            if [ -e "${teacher_dumpdir}/${train_set}/probs" ]; then
                # Knowledge distillation
                _feats_type=fbank
            else
                # Teacher forcing
                _feats_type="$(<${data_feats}/${train_set}/feats_type)"
            fi
        fi

        # NOTE(kamo): If feats_type=raw, vocoder_conf is unnecessary
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _fold_length="$((singing_fold_length * n_shift))"

        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        log "Generate '${svs_exp}/${inference_tag}/run.sh'. You can resume the process from stage 8 using this script"
        mkdir -p "${svs_exp}/${inference_tag}"; echo "${run_args} --stage 8 \"\$@\"; exit \$?" > "${svs_exp}/${inference_tag}/run.sh"; chmod +x "${svs_exp}/${inference_tag}/run.sh"


        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _speech_data="${_data}"
            _dir="${svs_exp}/${inference_tag}/${dset}"
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

            # Add X-vector to the inputs if needed
            if "${use_xvector}"; then
                _xvector_dir="${dumpdir}/xvector/${dset}"
                _ex_opts+="--data_path_and_name_and_type ${_xvector_dir}/xvector.scp,spembs,kaldi_ark "
            fi

            # Add spekaer ID to the inputs if needed
            if "${use_sid}"; then
                _ex_opts+="--data_path_and_name_and_type ${_data}/utt2sid,sids,text_int "
            fi

            # Add language ID to the inputs if needed
            if "${use_lid}"; then
                _ex_opts+="--data_path_and_name_and_type ${_data}/utt2lid,lids,text_int "
            fi

            _opts+="--discrete_token_layers ${discrete_token_layers} "
            _opts+="--mix_type ${mix_type} "


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

            # 3. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/svs_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/svs_inference.JOB.log \
                ${python} -m espnet2.bin.svs_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/text,text,text" \
                    --data_path_and_name_and_type "${_data}/label,label,duration" \
                    --data_path_and_name_and_type "${_data}/score.scp,score,score" \
                    --data_path_and_name_and_type "${_data}/${_scp},singing,${_type}" \
                    --data_path_and_name_and_type "${_data}/${token_file},discrete_token,text_int" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --model_file "${svs_exp}"/"${inference_model}" \
                    --train_config "${svs_exp}"/config.yaml \
                    --output_dir "${_logdir}"/output.JOB \
                    --vocoder_checkpoint "${vocoder_file}" \
                    ${_opts} ${_ex_opts} ${inference_args}

            # 4. Concatenates the output files from each jobs
            mkdir -p "${_dir}"/{norm,denorm,wav}

            if [ ${svs_task} == svs ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/norm/feats.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/norm/feats.scp"
                #for i in $(seq "${_nj}"); do
                #    cat "${_logdir}/output.${i}/denorm/feats.scp"
                #done | LC_ALL=C sort -k1 > "${_dir}/denorm/feats.scp"
            fi

            for i in $(seq "${_nj}"); do
                 cat "${_logdir}/output.${i}/speech_shape/speech_shape"
            done | LC_ALL=C sort -k1 > "${_dir}/speech_shape"
            for i in $(seq "${_nj}"); do
                mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
                rm -rf "${_logdir}/output.${i}"/wav
            done
            if [ -e "${_logdir}/output.${_nj}/att_ws" ]; then
                mkdir -p "${_dir}"/att_ws
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/durations/durations"
                done | LC_ALL=C sort -k1 > "${_dir}/durations"
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/focus_rates/focus_rates"
                done | LC_ALL=C sort -k1 > "${_dir}/focus_rates"
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/att_ws/*.png "${_dir}"/att_ws
                    rm -rf "${_logdir}/output.${i}"/att_ws
                done
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

    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Scoring"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _gt_wavscp="${_data}/wav.scp"
            _dir="${svs_exp}/${inference_tag}/${dset}"
            _gen_wavdir="${_dir}/wav"
            _gt_token_file="${_data}/${token_file}"
            _gen_token_scp="${_dir}/norm/feats.scp"

            # Objective Evaluation - Accuracy
            log "Begin Scoring for token accuracy on ${dset}, results are written under ${_dir}/Accuracy_res"

            mkdir -p "${_dir}/MCD_res"
            ${python} pyscripts/utils/evaluate_token_accuracy.py \
                ${_gen_token_scp} \
                ${_gt_token_file} \
                "${multi_token}" \
                --outdir "${_dir}/Accuracy_res" \
                --discrete_token_layers ${discrete_token_layers} \
                --mix_type ${mix_type}

            # Objective Evaluation - MCD
            log "Begin Scoring for MCD metrics on ${dset}, results are written under ${_dir}/MCD_res"

            mkdir -p "${_dir}/MCD_res"
            ${python} pyscripts/utils/evaluate_mcd.py \
                ${_gen_wavdir} \
                ${_gt_wavscp} \
                --outdir "${_dir}/MCD_res"

            # Objective Evaluation - log-F0 RMSE
            log "Begin Scoring for F0 related metrics on ${dset}, results are written under ${_dir}/F0_res"

            mkdir -p "${_dir}/F0_res"
            ${python} pyscripts/utils/evaluate_f0.py \
                ${_gen_wavdir} \
                ${_gt_wavscp} \
                --outdir "${_dir}/F0_res"

            # Objective Evaluation - semitone ACC
            log "Begin Scoring for SEMITONE related metrics on ${dset}, results are written under ${_dir}/SEMITONE_res"

            mkdir -p "${_dir}/SEMITONE_res"
            ${python} pyscripts/utils/evaluate_semitone.py \
                ${_gen_wavdir} \
                ${_gt_wavscp} \
                --outdir "${_dir}/SEMITONE_res"

             # Objective Evaluation - VUV error
            log "Begin Scoring for VUV related metrics on ${dset}, results are written under ${_dir}/VUV_res"

            mkdir -p "${_dir}/VUV_res"
            ${python} pyscripts/utils/evaluate_vuv.py \
                ${_gen_wavdir} \
                ${_gt_wavscp} \
                --outdir "${_dir}/VUV_res"

        done
    fi
else
    log "Skip the evaluation stages"
fi

packed_model="${svs_exp}/${svs_exp##*/}_${inference_model%.*}.zip"
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    log "Stage 10: Pack model: ${packed_model}"

    _opts=""
    if [ -e "${svs_stats_dir}/train/feats_stats.npz" ]; then
        _opts+=" --option ${svs_stats_dir}/train/feats_stats.npz"
    fi
    if [ -e "${svs_stats_dir}/train/pitch_stats.npz" ]; then
        _opts+=" --option ${svs_stats_dir}/train/pitch_stats.npz"
    fi
    if [ -e "${svs_stats_dir}/train/energy_stats.npz" ]; then
        _opts+=" --option ${svs_stats_dir}/train/energy_stats.npz"
    fi
    if "${use_xvector}"; then
        for dset in "${train_set}" ${test_sets}; do
            _opts+=" --option ${dumpdir}/xvector/${dset}/spk_xvector.scp"
            _opts+=" --option ${dumpdir}/xvector/${dset}/spk_xvector.ark"
        done
    fi
    if "${use_sid}"; then
        _opts+=" --option ${data_feats}/org/${train_set}/spk2sid"
    fi
    if "${use_lid}"; then
        _opts+=" --option ${data_feats}/org/${train_set}/lang2lid"
    fi
    ${python} -m espnet2.bin.pack svs \
        --train_config "${svs_exp}"/config.yaml \
        --model_file "${svs_exp}"/"${inference_model}" \
        --option "${svs_exp}"/images  \
        --outpath "${packed_model}" \
        ${_opts}

    # NOTE(kamo): If you'll use packed model to inference in this script, do as follows
    #   % unzip ${packed_model}
    #   % ./run.sh --stage 9 --svs_exp $(basename ${packed_model} .zip) --inference_model pretrain.pth
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
        # /some/where/espnet/egs2/foo/svs2/ -> foo/svs2
        _task="$(pwd | rev | cut -d/ -f1-2 | rev)"
        # foo/svs2 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${svs_exp}"/description
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
<li><strong>Config</strong><pre><code>$(cat "${svs_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${svs_exp}"/description \
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
        # /some/where/espnet/egs2/foo/svs2/ -> foo/svs2
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/svs2 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # copy files in ${dir_repo}
        unzip -o ${packed_model} -d ${dir_repo}
        # Generate description file
        # shellcheck disable=SC2034
        hf_task=singing-voice-synthesis
        # shellcheck disable=SC2034
        espnet_task=SVS
        # shellcheck disable=SC2034
        task_exp=${svs_exp}
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
