#!/usr/bin/env bash

# Copyright 2021 Peter Wu
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
skip_upload_hf=true  # Skip uploading to huggingface stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.
# TODO
backend=pytorch
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# Data preparation related
local_data_opts="" # Options to be passed to local/data.sh.

# Feature extraction related
feats_type=raw             # Input feature type.
audio_format=flac          # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
feats_extract=fbank        # On-the-fly feature extractor.
feats_normalize=global_mvn # On-the-fly feature normalizer.
fs=16000                   # Sampling rate.
n_fft=1024                 # The number of fft points.
n_shift=256                # The number of shift points.
win_length=null            # Window length.
fmin=80                    # Minimum frequency of Mel basis.
fmax=7600                  # Maximum frequency of Mel basis.
n_mels=80                  # The number of mel basis.

# Training related
train_config=""    # Config for training.
tag=""             # Suffix for training directory.
vc_stats_dir=""   # Specify the directory path for statistics. If empty, automatically decided.

# Decoding related
inference_config="" # Config for decoding.
inference_model=train.loss.ave.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth

# [Task dependent] Set the datadir name created by local/data.sh
src_train_set=""     # Name of src training set.
src_valid_set=""     # Name of src validation set used for monitoring/tuning network training.
src_test_sets=""     # Names of src test sets. Multiple items (e.g., both dev and eval sets) can be specified.
trg_train_set=""     # Name of src training set.
trg_valid_set=""     # Name of src validation set used for monitoring/tuning network training.
trg_test_sets=""     # Names of src test sets. Multiple items (e.g., both dev and eval sets) can be specified.

# TODO
# decoding related
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage
voc=PWG                     # vocoder used (GL or PWG)
griffin_lim_iters=64        # The number of iterations of Griffin-Lim

# TODO
# pretrained model related
pretrained_model=           # available pretrained models: m_ailabs.judy.vtn_tts_pt

 # TODO
# dataset configuration
db_root=downloads
srcspk=clb                  # available speakers: "slt" "clb" "bdl" "rms"
trgspk=slt
num_train_utts=-1           # -1: use all 932 utts
norm_name=                  # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization

# TODO
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
    --use_xvector      # Whether to use X-vector (Require Kaldi, default="${use_xvector}").
    --use_sid          # Whether to use speaker id as the inputs (default="${use_sid}").
    --use_lid          # Whether to use language id as the inputs (default="${use_lid}").
    --feats_extract    # On the fly feature extractor (default="${feats_extract}").
    --feats_normalize  # Feature normalizer for on the fly feature extractor (default="${feats_normalize}")
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
    --tts_exp       # Specify the directory path for experiment.
                    # If this option is specified, tag is ignored (default="${tts_exp}").
    --vc_stats_dir # Specify the directory path for statistics.
                    # If empty, automatically decided (default="${vc_stats_dir}").
    --num_splits    # Number of splitting for tts corpus (default="${num_splits}").
    --teacher_dumpdir       # Directory of teacher outputs (needed if tts=fastspeech, default="${teacher_dumpdir}").
    --write_collected_feats # Whether to dump features in statistics collection (default="${write_collected_feats}").
    --tts_task              # TTS task {tts or gan_tts} (default="${tts_task}").

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
    --speech_fold_length # Fold length for speech data (default="${speech_fold_length}").
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

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats="${dumpdir}/raw"
else
    log "${help_message}"
    log "Error: only supported: --feats_type raw"
    exit 2
fi

# TODO
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

# TODO
# The directory used for collect-stats mode
if [ -z "${vc_stats_dir}" ]; then
    vc_stats_dir="${expdir}/tts_stats_${feats_type}"
    if [ "${feats_extract}" != fbank ]; then
        vc_stats_dir+="_${feats_extract}"
    fi
    vc_stats_dir+="_${token_type}"
    if [ "${cleaner}" != none ]; then
        vc_stats_dir+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        vc_stats_dir+="_${g2p}"
    fi
fi
# The directory used for training commands
if [ -z "${tts_exp}" ]; then
    tts_exp="${expdir}/tts_${tag}"
fi


# ========================== Main stages start from here. ==========================

pair=${srcspk}_${trgspk}
pair_train_set=${pair}_train
pair_valid_set=${pair}_dev
pair_test_set=${pair}_eval

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/*"
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi


    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
        for dset in "${src_train_set}" "${src_valid_set}" "${trg_train_set}" "${trg_valid_set}" ${trg_test_sets}; do
            if [ "${dset}" = "${src_train_set}" ] || [ "${dset}" = "${src_valid_set}"  || [ "${dset}" = "${trg_train_set}"  || [ "${dset}" = "${trg_valid_set}" ]; then
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
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================



if ! "${skip_train}"; then
    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: TTS collect stats"

        compute-cmvn-stats scp:data/${src_train_set}/feats.scp data/${src_train_set}/cmvn.ark
        compute-cmvn-stats scp:data/${trg_train_set}/feats.scp data/${trg_train_set}/cmvn.ark
        src_cmvn=data/${src_train_set}/cmvn.ark
        trg_cmvn=data/${trg_train_set}/cmvn.ark
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: VC Training"
    fi
fi

# [WIP]

if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
fi
src_feat_tr_dir=${dumpdir}/${src_train_set}_${norm_name}; mkdir -p ${src_feat_tr_dir}
src_feat_dt_dir=${dumpdir}/${src_valid_set}_${norm_name}; mkdir -p ${src_feat_dt_dir}
src_feat_ev_dir=${dumpdir}/${src_test_sets}_${norm_name}; mkdir -p ${src_feat_ev_dir}
trg_feat_tr_dir=${dumpdir}/${trg_train_set}_${norm_name}; mkdir -p ${trg_feat_tr_dir}
trg_feat_dt_dir=${dumpdir}/${trg_valid_set}_${norm_name}; mkdir -p ${trg_feat_dt_dir}
trg_feat_ev_dir=${dumpdir}/${trg_test_sets}_${norm_name}; mkdir -p ${trg_feat_ev_dir}
pair_tr_dir=${dumpdir}/${pair_train_set}_${norm_name}; mkdir -p ${pair_tr_dir}
pair_dt_dir=${dumpdir}/${pair_valid_set}_${norm_name}; mkdir -p ${pair_dt_dir}
pair_ev_dir=${dumpdir}/${pair_test_set}_${norm_name}; mkdir -p ${pair_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then # TODO delete after integrating stats & dump
    echo "stage 1: Feature Generation"

    # dump features
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_train_set}/feats.scp ${src_cmvn} exp/dump_feats/${src_train_set}_${norm_name} ${src_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_valid_set}/feats.scp ${src_cmvn} exp/dump_feats/${src_valid_set}_${norm_name} ${src_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${src_test_sets}/feats.scp ${src_cmvn} exp/dump_feats/${src_test_sets}_${norm_name} ${src_feat_ev_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_train_set}/feats.scp ${trg_cmvn} exp/dump_feats/${trg_train_set}_${norm_name} ${trg_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_valid_set}/feats.scp ${trg_cmvn} exp/dump_feats/${trg_valid_set}_${norm_name} ${trg_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${trg_test_sets}/feats.scp ${trg_cmvn} exp/dump_feats/${trg_test_sets}_${norm_name} ${trg_feat_ev_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    # make dummy dict
    dict="data/dummy_dict/X.txt"
    mkdir -p ${dict%/*}
    echo "<unk> 1" > ${dict}

    # make json labels
    data2json.sh --feat ${src_feat_tr_dir}/feats.scp \
         data/${src_train_set} ${dict} > ${src_feat_tr_dir}/data.json
    data2json.sh --feat ${src_feat_dt_dir}/feats.scp \
         data/${src_valid_set} ${dict} > ${src_feat_dt_dir}/data.json
    data2json.sh --feat ${src_feat_ev_dir}/feats.scp \
         data/${src_test_sets} ${dict} > ${src_feat_ev_dir}/data.json
    data2json.sh --feat ${trg_feat_tr_dir}/feats.scp \
         data/${trg_train_set} ${dict} > ${trg_feat_tr_dir}/data.json
    data2json.sh --feat ${trg_feat_dt_dir}/feats.scp \
         data/${trg_valid_set} ${dict} > ${trg_feat_dt_dir}/data.json
    data2json.sh --feat ${trg_feat_ev_dir}/feats.scp \
         data/${trg_test_sets} ${dict} > ${trg_feat_ev_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Pair Json Data Preparation"

    # make pair json
    if [ ${num_train_utts} -ge 0 ]; then
        make_pair_json.py \
            --src-json ${src_feat_tr_dir}/data.json \
            --trg-json ${trg_feat_tr_dir}/data.json \
            -O ${pair_tr_dir}/data_n${num_train_utts}.json \
            --num_utts ${num_train_utts}
    else
        make_pair_json.py \
            --src-json ${src_feat_tr_dir}/data.json \
            --trg-json ${trg_feat_tr_dir}/data.json \
            -O ${pair_tr_dir}/data.json
    fi
    make_pair_json.py \
        --src-json ${src_feat_dt_dir}/data.json \
        --trg-json ${trg_feat_dt_dir}/data.json \
        -O ${pair_dt_dir}/data.json
    make_pair_json.py \
        --src-json ${src_feat_ev_dir}/data.json \
        --trg-json ${trg_feat_ev_dir}/data.json \
        -O ${pair_ev_dir}/data.json
fi

if [[ -z ${train_config} ]]; then
    echo "Please specify --train_config."
    exit 1
fi

# If pretrained model specified, add pretrained model info in config
if [ -n "${pretrained_model}" ]; then
    pretrained_model_path=$(find ${db_root}/${pretrained_model} -name "snapshot*" | head -n 1)
    train_config="$(change_yaml.py \
        -a enc-init="${pretrained_model_path}" \
        -a dec-init="${pretrained_model_path}" \
        -o "conf/$(basename "${train_config}" .yaml).${tag}.yaml" "${train_config}")"
fi
if [ -z ${tag} ]; then
    expname=${srcspk}_${trgspk}_${backend}_$(basename ${train_config%.*})
else
    expname=${srcspk}_${trgspk}_${backend}_${tag}
fi
expdir=exp/${expname}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: VC model training"

    mkdir -p ${expdir}
    if [ ${num_train_utts} -ge 0 ]; then
        tr_json=${pair_tr_dir}/data_n${num_train_utts}.json
    else
        tr_json=${pair_tr_dir}/data.json
    fi
    dt_json=${pair_dt_dir}/data.json

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ -z "${inference_model}" ]; then
    inference_model="$(find "${expdir}" -name "snapshot*" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
    inference_model=$(basename ${inference_model})
fi
outdir=${expdir}/outputs_${inference_model}_$(basename ${inference_config%.*})
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding and synthesis"

    echo "Decoding..."
    pids=() # initialize pids
    for name in ${pair_valid_set} ${pair_test_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}_${norm_name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${inference_model} \
                --config ${inference_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Synthesis..."

    pids=() # initialize pids
    for name in ${pair_valid_set} ${pair_test_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}

        # Normalization
        # If not using pretrained models statistics, use statistics of target speaker
        if [ -n "${pretrained_model}" ]; then
            trg_cmvn="$(find "${db_root}/${pretrained_model}" -name "cmvn.ark" -print0 | xargs -0 ls -t | head -n 1)"
        else
            trg_cmvn=data/${trg_train_set}/cmvn.ark
        fi
        apply-cmvn --norm-vars=true --reverse=true ${trg_cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp

        # GL
        if [ ${voc} = "GL" ]; then
            echo "Using Griffin-Lim phase recovery."
            convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                --iters ${griffin_lim_iters} \
                ${outdir}_denorm/${name} \
                ${outdir}_denorm/${name}/log \
                ${outdir}_denorm/${name}/wav
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Objective Evaluation"

    for name in ${pair_valid_set} ${pair_test_set}; do
        local/ob_eval/evaluate.sh --nj ${nj} \
            --db_root ${db_root} \
            --vocoder ${voc} \
            ${outdir} ${name}
    done
fi
