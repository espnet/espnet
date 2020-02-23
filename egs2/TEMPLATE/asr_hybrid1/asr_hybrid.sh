#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
help_message=

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
stage=1          # Processes starts from the specified stage.
stop_stage=22    # Processes is stopped at the specified stage.
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
nj=32            # The number of parallel jobs.
decode_nj=32     # The number of parallel jobs in decoding.
gpu_decode=false # Whether to perform gpu decoding.
dumpdir=dump     # Directory to dump features.
mfccdir=mfcc     # Directory to dump MFCC used for gmm traning
ali_dir=ali      # Directory to dump the aligment by final gmm
expdir=exp       # Directory to save experiments.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# GMM train related
mono_num_iters=40
mono_tot_gauss=1000

skip_tri1=false
tri1_num_iters=35
tri1_num_leaves=1800
tri1_tot_gauss=9000

skip_tri2=false
tri2_num_iters=35
tri2_num_leaves=1800
tri2_tot_gauss=9000

tri3_num_iters=35
tri3_num_leaves=1800
tri3_tot_gauss=9000
decode_gmm=false  # decode gmm models

# Cleaning stage related
cleanup=false
segmentation_opts="--min-segment-length 0.3 --min-new-segment-length 0.6"

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format (only in feats_type=raw).
fs=16k            # Sampling rate.

# ASR model related
asr_tag=    # Suffix to the result dir for asr model training.
asr_config= # Config for asr model training.
asr_args=   # Arguments for asr model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in asr config.
feats_normalize=global_mvn  # Normalizaton layer type

# Decoding related
decode_tag=    # Suffix to the result dir for decoding.
decode_config= # Config for decoding.
decode_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
               # Note that it will overwrite args in decode config.
decode_model=valid.acc.best.pth # ASR model path for decoding.
                                # e.g.
                                # decode_model=train.loss.best.pth
                                # decode_model=3epoch/model.pth
                                # decode_model=valid.acc.best.pth
                                # decode_model=valid.loss.ave.pth


# Kaldi "latgen-faster-mapped" related
acwt=0.10 # note: only really affects pruning (scoring is on lattices).
beam=13.0
lattice_beam=8.0
min_active=200
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
num_threads=1
scoring_opts="--min-lmwt 4 --max-lmwt 15"


# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
num_devsets=100  # The number
eval_sets=       # Names of evaluation sets. Multiple items can be specified.
lang=data/lang


log "$0 $*"
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
[ -z "${eval_sets}" ] && { log "${help_message}"; log "Error: --eval_sets is required"; exit 2; };


# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi


# Set tag for naming of model directory
if [ -z "${asr_tag}" ]; then
    if [ -n "${asr_config}" ]; then
        asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
    else
        asr_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${asr_args}" ]; then
        asr_tag+="$(echo "${asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
# The directory used for collect-stats mode
asr_stats_dir="${expdir}/asr_stats"
# The directory used for training commands
asr_exp="${expdir}/asr_${asr_tag}"


# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${dev_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi


# ========================== [Require Kaldi] GMM-HMM stages ==========================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Make MFCC"

    # Extract MFCC here.
    # Note that these files are only used for gmm training and alignment and
    # they are not used at DNN stage any more.
    # In typical DNN-HMM training, we use more high resolution MFCC or FBANK features.
    for dset in ${train_set} ${eval_sets}; do
        # 1. Copy datadir
        utils/copy_data_dir.sh data/"${dset}" "${mfccdir}/${dset}"

        # 2. MFCC extract
        steps/make_mfcc.sh --nj "$(min ${nj} $(wc -l <${mfccdir}/${dset}/utt2spk))" \
            --cmd "${train_cmd}" "${mfccdir}/${dset}"

        # 3. CMVN stats
        steps/compute_cmvn_stats.sh "${mfccdir}/${dset}"
        utils/fix_data_dir.sh "${mfccdir}/${dset}"
    done

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Train mono"

    steps/train_mono.sh --nj "$(min ${nj} $(wc -l <${mfccdir}/${train_set}/spk2utt))" \
        --cmd "${train_cmd}" \
        --num_iters ${mono_num_iters} \
        --totgauss "${mono_tot_gauss}" \
        "${mfccdir}/${train_set}" "${lang}" exp/mono

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    if "${decode_gmm}"; then
        log "Stage 4: Decode mono"
        utils/mkgraph.sh "${lang}" exp/mono exp/mono/graph
        for dset in ${eval_sets}; do
            steps/decode.sh --config conf/decode.config \
                --nj "$(min ${nj} $(wc -l <${mfccdir}/${dset}/spk2utt))" --cmd "${decode_cmd}" \
                exp/mono/graph ${mfccdir}/test exp/mono/decode
        done
    else
        log "Stage 4: Skip decode mono"
    fi
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Align mono"

    steps/align_si.sh --nj "$(min ${nj} $(wc -l <${mfccdir}/${train_set}/spk2utt))" \
      --cmd "${train_cmd}" "${mfccdir}/${train_set}" "${lang}" exp/mono exp/mono_ali

fi


if ! "${skip_tri1}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Train tri1"

        steps/train_deltas.sh --cmd "${train_cmd}" \
            --num_iters ${tri1_num_iters} \
            "${tri1_num_leaves}" "${tri1_tot_gauss}" \
            "${mfccdir}/${train_set}" "${lang}" exp/mono_ali exp/tri1

    fi


    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        if "${decode_gmm}"; then
            log "Stage 7: Decode tri1"

            utils/mkgraph.sh "${lang}" exp/tri1 exp/tri1/graph
            for dset in ${eval_sets}; do
                steps/decode.sh --config conf/decode.config \
                    --nj "$(min ${nj} $(wc -l <${mfccdir}/${dset}/spk2utt))" --cmd "${decode_cmd}" \
                    exp/tri1/graph "${mfccdir}/${dset}" exp/tri1/"decode_${dset}"
                cat "exp/tri1/decode_${dset}/scoring_kaldi/best_wer"
            done
        else
            log "Stage 7: Skip decode tri1"
        fi
    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Align tri1"

        steps/align_si.sh --nj "$(min ${nj} $(wc -l <${mfccdir}/${train_set}/spk2utt))" \
            --cmd "${train_cmd}" \
            "${mfccdir}/${train_set}" "${lang}" exp/tri1 exp/tri1_ali
    fi
else
    log "Skip 6-8 tri1 stages"
fi

if "${skip_tri1}"; then
    gmm_dir=exp/mono
    next_ali=exp/mono_ali
else
    gmm_dir=exp/tri1
    next_ali=exp/tri1_ali
fi

if ! "${skip_tri2}"; then
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Train tri2 (LDA+MLLT)"

        steps/train_lda_mllt.sh --cmd "${train_cmd}" \
            --num_iters ${tri2_num_iters} \
            "${tri2_num_leaves}" "${tri2_tot_gauss}" \
            "${mfccdir}/${train_set}" "${lang}" "${next_ali}" exp/tri2

    fi


    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        if "${decode_gmm}"; then
            log "Stage 10: Decode tri2"

            utils/mkgraph.sh "${lang}" exp/tri2 exp/tri2/graph
            for dset in ${eval_sets}; do
                steps/decode.sh --config conf/decode.config \
                    --nj "$(min ${nj} $(wc -l <${mfccdir}/${dset}/spk2utt))" --cmd "${decode_cmd}" \
                    exp/tri2/graph "${mfccdir}/${dset}" exp/tri2/"decode_${dset}"
                cat "exp/tri2/decode_${dset}/scoring_kaldi/best_wer"
            done
        else
            log "Stage 10: Skip decode tri2"
        fi
    fi


    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        log "Stage 11: Align tri2"

        steps/align_si.sh --nj "$(min ${nj} $(wc -l <${mfccdir}/${train_set}/spk2utt))" \
            --cmd "${train_cmd}" \
            "${mfccdir}/${train_set}" "${lang}" exp/tri2 exp/tri2_ali
    fi


else
    log "Skip 9-11 tri2 stages"
fi

if "${skip_tri2}"; then
    gmm_dir="${gmm_dir}"
    next_ali="${next_ali}"
else
    gmm_dir=exp/tri2
    next_ali=exp/tri2_ali
fi


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    log "Stage 12: Train tri3 (LDA+MLLT+SAT)"

    steps/train_sat.sh --cmd "${train_cmd}" \
        --num_iters ${tri3_num_iters} \
        "${tri3_num_leaves}" "${tri3_tot_gauss}" \
        "${mfccdir}/${train_set}" "${lang}" "${next_ali}" exp/tri3

fi


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    utils/mkgraph.sh "${lang}" exp/tri3 exp/tri3/graph

    if "${decode_gmm}"; then
        log "Stage 13: Decode tri3"
        for dset in ${eval_sets}; do
            steps/decode_fmllr.sh --config conf/decode.config \
                --nj "$(min ${nj} $(wc -l <${mfccdir}/${dset}/spk2utt))" --cmd "${decode_cmd}" \
                exp/tri3/graph "${mfccdir}/${dset}" exp/tri3/"decode_${dset}"
            cat "exp/tri3/decode_${dset}/scoring_kaldi/best_wer"
        done
    else
        log "Stage 13: Skip decode tri3"
    fi
fi


gmm_dir=exp/tri3
next_ali=exp/tri3_ali


if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    if "${cleanup}"; then
        log "Stage 14: Perform data cleanup for training data"

        if [ ! -f "${mfccdir}/${train_set}/segments" ]; then
            log "Error: ${mfccdir}/${train_set}/segments doesn't exist"
            exit 1
        fi

        steps/cleanup/clean_and_segment_data.sh \
            --nj "$(min ${nj} $(wc -l <${mfccdir}/${train_set}/spk2utt))" --cmd "${train_cmd}" \
            --segmentation-opts "${segmentation_opts}" \
            "${mfccdir}/${train_set}" "${lang}" exp/tri3 exp/tri3_cleaned "${mfccdir}/${train_set}_cleaned"
        utils/mkgraph.sh "${lang}" exp/tri3_cleaned exp/tri3_cleaned/graph

    else
        log "Stage 14: Skip data cleanup"
    fi
fi


# ========================== Prepare data for DNN ==========================
if "${cleanup}"; then
    gmm_dir="${gmm_dir}_cleaned"
    train_set="${train_set}_cleaned"
    train_nodev_set="train_nodev_cleaned"
    dev_set="dev_cleaned"
else
    gmm_dir="${gmm_dir}"
    train_set="${train_set}"
    train_nodev_set="train_nodev"
    dev_set="dev"
fi


if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    log "Stage15: Split ${train_set} into train_nodev and dev"

    utils/subset_data_dir.sh --first \
        "${mfccdir}/${train_set}" "${num_devsets}" "${mfccdir}/${dev_set}"
    n=$(($(wc -l < ${mfccdir}/${train_set}/text) - num_devsets))
    utils/subset_data_dir.sh --last \
        "${mfccdir}/${train_set}" "${n}" "${mfccdir}/${train_nodev_set}"

    for dset in ${train_nodev_set} ${dev_set}; do
        utils/copy_data_dir.sh "${mfccdir}/${dset}" "data/${dset}"
        rm -f data/"${dset}"/{feats.scp,cmvn.scp,utt2num_frames,utt2max_frames}
    done
fi


if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    log "Stage 16: Align tri3 and convert aligments to pdfs-ids"

    for dset in ${train_nodev_set} ${dev_set}; do
        steps/align_fmllr.sh --nj "$(min ${nj} $(wc -l <${mfccdir}/${dset}/spk2utt))" \
            --cmd "${train_cmd}" \
            "${mfccdir}/${dset}" "${lang}" "${gmm_dir}" "${ali_dir}/${dset}/ali"

        # NOTE(kamo):
        # pdfs.txt is exactly what we use as the target of DNN training.
        # About Kaldi's HMM topologies: https://kaldi-asr.org/doc/hmm.html
        ali-to-pdf "${gmm_dir}"/final.mdl \
            ark:"gunzip -c ${ali_dir}/${dset}/ali/ali.*.gz|" ark,t:"${ali_dir}/${dset}/pdfs.txt"
    done
    hmm-info "${gmm_dir}"/final.mdl | grep "number of pdfs" | awk '{ print $4 }' > "${ali_dir}"/num_targets

fi


if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    if [ -n "${speed_perturb_factors}" ]; then
       log "Stage l7: Speed perturbation: data/${train_set} -> data/${train_set}_sp"

       for factor in ${speed_perturb_factors}; do
           scripts/utils/perturb_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
           _dirs+="data/${train_set}_sp${factor} "
       done

       utils/combine_data.sh "data/${train_set}" ${_dirs}
    else
       log "Skip stage 16: Speed perturbation"
    fi
fi

if [ -n "${speed_perturb_factors}" ]; then
    train_set="${train_set}_sp"
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
    if [ "${feats_type}" = raw ]; then
        log "Stage 18: Format wav.scp: data/ -> ${data_feats}/"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in "${train_nodev_set}" "${dev_set}" ${eval_sets}; do
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}/${dset}"
            rm -f ${data_feats}/${dset}/{segments,wav.scp,reco2file_and_channel}
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
                "data/${dset}/wav.scp" "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
        done

    elif [ "${feats_type}" = fbank_pitch ]; then
        log "[Require Kaldi] Stage 17: ${feats_type} extract: data/ -> ${data_feats}/"

        for dset in "${train_nodev_set}" "${dev_set}" ${eval_sets}; do
            # 1. Copy datadir
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}/${dset}"

            # 2. Feature extract
            _nj=$(min "${nj}" "$(<"${data_feats}/${dset}/utt2spk" wc -l)")
            steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}/${dset}"
            utils/fix_data_dir.sh "${data_feats}/${dset}"

            # 3. Derive the the frame length and feature dimension
            scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                "${data_feats}/${dset}/feats.scp" "${data_feats}/${dset}/feats_shape"

            # 4. Write feats_dim
            head -n 1 "${data_feats}/${dset}/feats_shape" | awk '{ print $2 }' \
                | cut -d, -f2 > ${data_feats}/${dset}/feats_dim

            # 5. Write feats_type
            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
        done

    elif [ "${feats_type}" = fbank ]; then
        log "Stage 17: ${feats_type} extract: data/ -> ${data_feats}/"
        log "${feats_type} is not supported yet."
        exit 1

    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
fi

# ========================== Data preparation is done here. ==========================


if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
    _asr_train_dir="${data_feats}/${train_nodev_set}"
    _asr_dev_dir="${data_feats}/${dev_set}"
    log "Stage 19: ASR collect stats: train_set=${_asr_train_dir}, dev_set=${_asr_dev_dir}"

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
        _type=sound
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
    _nj=$(min "${nj}" "$(<${_asr_train_dir}/${_scp} wc -l)" "$(<${_asr_dev_dir}/${_scp} wc -l)")

    key_file="${_asr_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_asr_dev_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/dev.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # FIXME(kamo): max_length is confusing name. How about fold_length?

    # 2. Submit jobs
    log "ASR collect-stats started... log: '${_logdir}/stats.*.log'"

    # NOTE: --*_shape_file doesn't require length information if --batch_type=const --sort_in_batch=none

    # shellcheck disable=SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python3 -m espnet2.bin.asr_hybrid_train \
            --collect_stats true \
            --iterator_type sequence \
            --batch_type const_no_sort \
            --num_targets "$(<${ali_dir}/num_targets)" \
            --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
            --train_data_path_and_name_and_type "${ali_dir}/${train_nodev_set}/pdfs.txt,align,text_int" \
            --valid_data_path_and_name_and_type "${_asr_dev_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${ali_dir}/${dev_set}/pdfs.txt,align,text_int" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/dev.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${asr_args}

    # 3. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${asr_stats_dir}"
fi


if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
    _asr_train_dir="${data_feats}/${train_nodev_set}"
    _asr_dev_dir="${data_feats}/${dev_set}"
    log "Stage 20: ASR Training: train_set=${_asr_train_dir}, dev_set=${_asr_dev_dir}"

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
        _type=sound
        _max_length=80000
        _opts+="--frontend_conf fs=${fs} "
    else
        _scp=feats.scp
        _type=kaldi_ark
        _max_length=800
        _input_size="$(<${_asr_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "

    fi
    if [ "${feats_normalize}" = global_mvn ]; then
        # Default normalization is utterance_mvn and changes to global_mvn
        _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz"
    fi

    log "ASR training started... log: '${asr_exp}/train.log'"
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.launch \
        --cmd "${cuda_cmd}" \
        --log "${asr_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${asr_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        python3 -m espnet2.bin.asr_hybrid_train \
            --num_targets "$(<${ali_dir}/num_targets)" \
            --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
            --train_data_path_and_name_and_type "${ali_dir}/${train_nodev_set}/pdfs.txt,align,text_int" \
            --valid_data_path_and_name_and_type "${_asr_dev_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${ali_dir}/${dev_set}/pdfs.txt,align,text_int" \
            --train_shape_file "${asr_stats_dir}/train/speech_shape" \
            --train_shape_file "${asr_stats_dir}/train/align_shape" \
            --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${asr_stats_dir}/valid/align_shape" \
            --resume true \
            --max_length "${_max_length}" \
            --max_length 150 \
            --output_dir "${asr_exp}" \
            ${_opts} ${asr_args}

fi


if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ]; then
    log "Stage 21: Decoding: training_dir=${asr_exp}"

    if ${gpu_decode}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
        _ngpu=0
    fi

    _opts=
    if [ -n "${decode_config}" ]; then
        _opts+="--config ${decode_config} "
    fi
    if [ "${num_threads}" -gt 1 ]; then
        thread_string="-parallel --num-threads=${num_threads}"
    else
        thread_string=""
    fi

    for dset in "${dev_set}" ${eval_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${asr_exp}/decode_${dset}${decode_tag}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        _feats_type="$(<${_data}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            _type=sound
        else
            _scp=feats.scp
            _type=kaldi_ark
        fi

        # 1. Split the key file
        key_file=${_data}/${_scp}
        split_scps=""
        _nj=$(min "${decode_nj}" "$(<${key_file} wc -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/asr_decode.*.log'"
        # shellcheck disable=SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_decode.JOB.log \
            python3 -m espnet2.bin.asr_hybrid_decode \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                --pdfs "${ali_dir}/${train_nodev_set}/pdfs.txt" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --train_config "${asr_exp}"/config.yaml \
                --model_file "${asr_exp}"/"${decode_model}" \
                --wspecifier ark:- \
                ${_opts} ${decode_args} \| \
            latgen-faster-mapped"${thread_string}" \
                --min-active="${min_active}" \
                --max-active="${max_active}" \
                --max-mem="${max_mem}" \
                --beam="${beam}" \
                --lattice-beam="${lattice_beam}" \
                --acoustic-scale="${acwt}" \
                --allow-partial=true \
                --word-symbol-table="${gmm_dir}"/graph/words.txt \
                "${gmm_dir}/final.mdl" \
                "${gmm_dir}"/graph/HCLG.fst ark:- "ark:|gzip -c > ${_logdir}/lat.JOB.gz" || exit 1;

    done
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ]; then
    log "Stage 22: Scoring"

    for dset in "${dev_set}" ${eval_sets}; do
        # shellcheck disable=SC2086
        steps/scoring/score_kaldi_wer.sh ${scoring_opts} --cmd "${decode_cmd}"  \
            "${data_feats}/${dset}" \
            "${gmm_dir}"/graph \
            "${asr_exp}/decode_${dset}${decode_tag}/logdir"

        # shellcheck disable=SC2086
        steps/scoring/score_kaldi_cer.sh --stage 2 ${scoring_opts} --cmd "${decode_cmd}" \
            "${data_feats}/${dset}" \
            "${gmm_dir}"/graph \
            "${asr_exp}/decode_${dset}${decode_tag}/logdir"
    done
fi


if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ]; then
    log "[Option] Stage 23: Pack model: ${asr_exp}/packed.tgz"

    # shellcheck disable=SC2086
    python -m espnet2.bin.pack asr \
        --asr_train_config.yaml "${asr_exp}"/config.yaml \
        --asr_model_file.pth "${asr_exp}"/"${decode_model}" \
        --option "${asr_exp}"/RESULTS.md \
        --outpath "${asr_exp}/packed.tgz"

fi

log "Successfully finished. [elapsed=${SECONDS}s]"

