#!/usr/bin/env bash

# Speech-tokenizer recipe template.
#
# Stages 1--5 contain the standard ASR data preparation pipeline.  Stage 6
# extracts one speaker embedding per utterance for reconstruction conditioning.

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
    local value=$1
    local candidate
    for candidate in "$@"; do
        if [ "${candidate}" -le "${value}" ]; then
            value=${candidate}
        fi
    done
    echo "${value}"
}

SECONDS=0

# General configuration
stage=1
stop_stage=10000
skip_stages=
ngpu=1
num_nodes=1
nj=32
inference_nj=32
gpu_inference=false
dumpdir=dump
expdir=exp
python=python3

# Data preparation (stages 1--5)
local_data_opts=
speed_perturb_factors=
feats_type=raw
audio_format=flac
fs=16k
min_wav_duration=0.1
max_wav_duration=20
multi_columns_input_wav_scp=false
multi_columns_output_wav_scp=false
post_process_local_data_opts=
token_type=bpe
nbpe=5000
bpemode=unigram
bpe_input_sentence_size=100000000
bpe_char_cover=1.0
bpe_nlsyms=
nlsyms_txt=none
cleaner=none
g2p=none
lang=noinfo
blank="<blank>"
oov="<unk>"
sos_eos="<sos/eos>"

# Speaker embedding extraction (stage 6)
spk_embed_tool=espnet
spk_embed_model=espnet/voxcelebs12_ecapa_wavlm_joint
spk_embed_tag=espnet_spk
spk_embed_gpu_inference=true
spk_embed_parallel=true
spk_embed_num_workers=4
spk_embed_batch_size=8
spk_embed_prefetch=128

# Offline K-means initialization (stage 7)
use_default_centroid=true
kmeans_feature=wavlm_large/21
nclusters=2000
kmeans_portion=0.1
kmeans_storage_save_mode=true
kmeans_use_gpu=true
kmeans_num_threads=20
kmeans_opts=

# Experiment configuration (used by later stages)
tok_task=gan_tok  # gan_tok or tok
tok_tag=
tok_exp=
tok_stats_dir=
tok_config=
tok_args=
inference_model=valid.loss.ave.pth
inference_tag=  # Output tag. If empty, derived from inference_model.
inference_batch_size=1

# Task-dependent configuration
train_set=
valid_set=
test_sets=
bpe_train_text=

help_message=$(cat << EOF
Usage: $0 --train_set <train_set> --valid_set <valid_set> --test_sets "<test sets>" [options]

Stages:
  1-5: Standard ASR data preparation, waveform formatting, filtering, and token list
    6: Utterance-level speaker embedding extraction
    7: SSL feature extraction and offline K-means centroid initialization
    8: Collect speech, text, and speaker-embedding shapes
    9: Joint ASR and adversarial reconstruction training
   10: Tokenizer-only inference

Main options:
  --stage INT
  --stop_stage INT
  --skip_stages "INT ..."
  --train_set NAME
  --valid_set NAME
  --test_sets "NAME ..."
  --tok_config PATH
  --tok_task gan_tok|tok
  --spk_embed_model TAG_OR_PATH
  --spk_embed_gpu_inference true|false
  --use_default_centroid true|false
  --kmeans_feature MODEL/LAYER
  --nclusters INT
  --kmeans_portion FLOAT

The default speaker encoder is espnet/voxcelebs12_ecapa_wavlm_joint.  Stage 6
writes per-utterance embeddings to
  ${dumpdir}/${spk_embed_tag}/<set>/${spk_embed_tag}.scp
for use as "spembs,kaldi_ark".  Speaker-level averaged embeddings are not used.

If --use_default_centroid is true, stage 7 trains K-means and its output path
is passed to the quantizer at stage 9.  If false, stage 7 is skipped and no
centroid path is passed on the command line; quantizer_conf.centroid_path must
then be provided by --tok_config.
EOF
)

if [ -f ./path.sh ]; then
    # shellcheck disable=SC1091
    . ./path.sh
fi
if [ -f ./cmd.sh ]; then
    # shellcheck disable=SC1091
    . ./cmd.sh
fi

if [ $# -eq 0 ]; then
    echo "${help_message}"
    exit 2
fi

# shellcheck disable=SC1091
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are supported: $*"
    echo "${help_message}"
    exit 2
fi
if [ -z "${train_set}" ] || [ -z "${valid_set}" ]; then
    log "Error: --train_set and --valid_set are required"
    echo "${help_message}"
    exit 2
fi
if [ "${feats_type}" != raw ]; then
    log "Error: tok1 currently supports --feats_type raw only"
    exit 2
fi
if [ "${tok_task}" != gan_tok ] && [ "${tok_task}" != tok ]; then
    log "Error: --tok_task must be gan_tok or tok"
    exit 2
fi
if [ "${use_default_centroid}" != true ] && [ "${use_default_centroid}" != false ]; then
    log "Error: --use_default_centroid must be true or false"
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
spk_embed_script="${script_dir}/pyscripts/utils/extract_spk_embed_parallel.py"
spk_embed_script_serial="${script_dir}/pyscripts/utils/extract_spk_embed.py"
kmeans_script="${script_dir}/scripts/feats/perform_kmeans.sh"

# Derived data and token-list paths follow the ASR recipe convention.
effective_train_set=${train_set}
if [ -n "${speed_perturb_factors}" ]; then
    effective_train_set=${train_set}_sp
fi
data_feats=${dumpdir}/raw
kmeans_feature_type=${kmeans_feature%%/*}
if [ "${kmeans_feature}" = mfcc ]; then
    kmeans_layer=
    kmeans_feature_conf="{type=mfcc}"
else
    if [ "${kmeans_feature}" = "${kmeans_feature_type}" ]; then
        log "Error: --kmeans_feature must have MODEL/LAYER format: ${kmeans_feature}"
        exit 2
    fi
    kmeans_layer=${kmeans_feature#*/}
    kmeans_feature_conf="{type=s3prl,conf={s3prl_conf={upstream=${kmeans_feature_type}},download_dir=hub,multilayer_feature=False,layer=${kmeans_layer}}}"
fi
kmeans_dir=${expdir}/kmeans/$(echo "${kmeans_feature}" | tr / _)_${nclusters}clusters
if "${use_default_centroid}"; then
    resolved_centroid_path=${kmeans_dir}/km_${nclusters}.mdl
fi
if [ -z "${tok_tag}" ]; then
    if [ -n "${tok_config}" ]; then
        tok_tag=${tok_task}_$(basename "${tok_config}" .yaml)_${feats_type}
    else
        tok_tag=${tok_task}_train_${feats_type}
    fi
    if [ "${lang}" != noinfo ]; then
        tok_tag+=_${lang}_${token_type}
    else
        tok_tag+=_${token_type}
    fi
    if [ "${token_type}" = bpe ]; then
        tok_tag+=${nbpe}
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        tok_tag+=_sp
    fi
fi
if [ -z "${tok_stats_dir}" ]; then
    # Stats depend on the input data and preprocessing, not on the training
    # config or K-means initializer.  Keep this human-readable, following the
    # ASR recipe convention, while distinguishing the additional speaker input.
    stats_token_tag=${lang}_${token_type}
    if [ "${token_type}" = bpe ]; then
        stats_token_tag+=_${bpemode}${nbpe}
    fi
    stats_speed_tag=
    if [ -n "${speed_perturb_factors}" ]; then
        stats_speed_tag=_sp$(echo "${speed_perturb_factors}" | tr ' ' '-')
    fi
    stats_spemb_tag=${spk_embed_tool}_${spk_embed_model}_${spk_embed_tag}
    stats_spemb_tag=$(echo "${stats_spemb_tag}" | sed -e 's#[/ :,]#_#g')
    tok_stats_dir=${expdir}/tok_stats_\
${effective_train_set}_${valid_set}_${feats_type}_${fs}_${audio_format}_\
${stats_token_tag}${stats_speed_tag}_${stats_spemb_tag}
fi
if [ -z "${tok_exp}" ]; then
    tok_exp=${expdir}/tok_${tok_tag}
fi
if [ -z "${inference_tag}" ]; then
    inference_tag="decode_model_$(echo "${inference_model}" \
        | sed -e 's#/#_#g' -e 's/\.[^.]*$//g')"
fi
if [ "${token_type}" = bpe ]; then
    token_list=data/${lang}_token_list/bpe_${bpemode}${nbpe}/tokens.txt
    bpedir=data/${lang}_token_list/bpe_${bpemode}${nbpe}
    bpeprefix=${bpedir}/bpe
    bpemodel=${bpeprefix}.model
elif [ "${token_type}" = char ]; then
    token_list=data/${lang}_token_list/char/tokens.txt
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list=data/${lang}_token_list/word/tokens.txt
    bpemodel=none
else
    log "Error: unsupported --token_type ${token_type}; use bpe, char, or word"
    exit 2
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "Stage 1: Data preparation"
    # [Task dependent] Each corpus recipe provides local/data.sh.
    # shellcheck disable=SC2086
    local/data.sh ${local_data_opts}
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    if [ -n "${speed_perturb_factors}" ]; then
        log "Stage 2: Speed perturbation: data/${train_set} -> data/${effective_train_set}"
        perturb_dirs=()
        for factor in ${speed_perturb_factors}; do
            if "${python}" -c "assert ${factor} != 1.0" 2>/dev/null; then
                scripts/utils/perturb_data_dir_speed.sh \
                    "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
                perturb_dirs+=("data/${train_set}_sp${factor}")
            else
                perturb_dirs+=("data/${train_set}")
            fi
        done
        utils/combine_data.sh "data/${effective_train_set}" "${perturb_dirs[@]}"
    else
        log "Skip stage 2: No speed perturbation factors were specified"
    fi
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    log "Stage 3: Format wav.scp: data/ -> ${data_feats}"
    format_sets=("${effective_train_set}")
    if [ "${effective_train_set}" != "${train_set}" ]; then
        # Keep an unperturbed formatted copy for offline K-means initialization.
        format_sets+=("${train_set}")
    fi
    format_sets+=("${valid_set}")
    # shellcheck disable=SC2206
    format_sets+=(${test_sets})
    for dset in "${format_sets[@]}"; do
        if [ "${dset}" = "${effective_train_set}" ] \
                || [ "${dset}" = "${train_set}" ] \
                || [ "${dset}" = "${valid_set}" ]; then
            suffix=/org
        else
            suffix=
        fi
        target_dir=${data_feats}${suffix}/${dset}
        utils/copy_data_dir.sh --validate_opts --non-print "data/${dset}" "${target_dir}"
        rm -f "${target_dir}/segments" "${target_dir}/wav.scp" \
            "${target_dir}/reco2file_and_channel" "${target_dir}/reco2dur"

        format_opts=()
        if [ -e "data/${dset}/segments" ]; then
            format_opts+=(--segments "data/${dset}/segments")
        fi
        scripts/audio/format_wav_scp.sh \
            --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" \
            --multi-columns-input "${multi_columns_input_wav_scp}" \
            --multi-columns-output "${multi_columns_output_wav_scp}" \
            "${format_opts[@]}" \
            "data/${dset}/wav.scp" "${target_dir}"

        echo raw > "${target_dir}/feats_type"
        if "${multi_columns_output_wav_scp}"; then
            echo "multi_${audio_format}" > "${target_dir}/audio_format"
        else
            echo "${audio_format}" > "${target_dir}/audio_format"
        fi
    done
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"
    filter_sets=("${effective_train_set}")
    if [ "${effective_train_set}" != "${train_set}" ]; then
        filter_sets+=("${train_set}")
    fi
    filter_sets+=("${valid_set}")
    for dset in "${filter_sets[@]}"; do
        source_dir=${data_feats}/org/${dset}
        target_dir=${data_feats}/${dset}
        utils/copy_data_dir.sh --validate_opts --non-print "${source_dir}" "${target_dir}"
        cp "${source_dir}/feats_type" "${target_dir}/feats_type"

        sample_rate=$("${python}" -c "import humanfriendly as h; print(h.parse_size('${fs}'))")
        min_length=$("${python}" -c "print(int(${min_wav_duration} * ${sample_rate}))")
        max_length=$("${python}" -c "print(int(${max_wav_duration} * ${sample_rate}))")
        awk -v min_length="${min_length}" -v max_length="${max_length}" \
            '{ if ($2 > min_length && $2 < max_length) print $0; }' \
            "${source_dir}/utt2num_samples" > "${target_dir}/utt2num_samples"
        utils/filter_scp.pl "${target_dir}/utt2num_samples" \
            < "${source_dir}/wav.scp" > "${target_dir}/wav.scp"
        awk '{ if (NF != 1) print $0; }' \
            "${source_dir}/text" > "${target_dir}/text"
        utils/fix_data_dir.sh "${target_dir}"
    done

    if [ -n "${post_process_local_data_opts}" ]; then
        # shellcheck disable=SC2086
        local/data.sh ${post_process_local_data_opts} \
            --asr_data_dir "${data_feats}/${effective_train_set}"
    fi
    awk '{ if (NF != 1) print $0; }' \
        "${data_feats}/${effective_train_set}/text" > "${data_feats}/lm_train.txt"
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    mkdir -p "$(dirname "${token_list}")"
    if [ "${token_type}" = bpe ]; then
        if [ -z "${bpe_train_text}" ]; then
            bpe_train_text=${data_feats}/lm_train.txt
        fi
        log "Stage 5: Generate BPE token list from ${bpe_train_text}"
        cut -f 2- -d " " "${bpe_train_text}" > "${bpedir}/train.txt"
        spm_opts=()
        if [ -n "${bpe_nlsyms}" ]; then
            if [ -f "${bpe_nlsyms}" ]; then
                bpe_nlsyms_list=$(awk '{print $1}' "${bpe_nlsyms}" | paste -s -d, -)
                spm_opts+=("--user_defined_symbols=${bpe_nlsyms_list}")
            else
                spm_opts+=("--user_defined_symbols=${bpe_nlsyms}")
            fi
        fi
        spm_train \
            --input="${bpedir}/train.txt" \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpeprefix}" \
            --character_coverage="${bpe_char_cover}" \
            --input_sentence_size="${bpe_input_sentence_size}" \
            "${spm_opts[@]}"
        {
            echo "${blank}"
            echo "${oov}"
            awk 'NR != 1 && NR != 2 && NR != 3 {print $1}' "${bpeprefix}.vocab"
            echo "${sos_eos}"
        } > "${token_list}"
    else
        log "Stage 5: Generate ${token_type} token list"
        "${python}" -m espnet2.bin.tokenize_text \
            --token_type "${token_type}" \
            --input "${data_feats}/lm_train.txt" \
            --output "${token_list}" \
            --field 2- \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"
    fi
fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]6[[:space:]] ]]; then
    log "Stage 6: Extract utterance-level speaker embeddings"

    if "${spk_embed_gpu_inference}"; then
        spk_cmd=${cuda_cmd:-run.pl}
        spk_ngpu=1
        spk_device=cuda
    else
        spk_cmd=${decode_cmd:-run.pl}
        spk_ngpu=0
        spk_device=cpu
    fi

    if "${spk_embed_parallel}"; then
        extractor=${spk_embed_script}
        extractor_opts=(
            --num_workers "${spk_embed_num_workers}"
            --batch_size "${spk_embed_batch_size}"
            --prefetch "${spk_embed_prefetch}"
        )
    else
        extractor=${spk_embed_script_serial}
        extractor_opts=()
    fi
    if [ ! -f "${extractor}" ]; then
        log "Error: Speaker embedding extractor was not found: ${extractor}"
        exit 1
    fi

    for dset in "${effective_train_set}" "${valid_set}" ${test_sets}; do
        # Use stage 4's filtered train/validation data and stage 3's test data.
        data_dir="${dumpdir}/raw/${dset}"
        output_dir="${dumpdir}/${spk_embed_tag}/${dset}"
        mkdir -p "${output_dir}"
        if [ ! -f "${data_dir}/wav.scp" ]; then
            log "Error: Missing stage 3 output: ${data_dir}/wav.scp"
            exit 1
        fi

        "${spk_cmd}" --gpu "${spk_ngpu}" "${output_dir}/spk_embed_extract.log" \
            "${python}" "${extractor}" \
                --pretrained_model "${spk_embed_model}" \
                --toolkit "${spk_embed_tool}" \
                --spk_embed_tag "${spk_embed_tag}" \
                --device "${spk_device}" \
                "${extractor_opts[@]}" \
                "${data_dir}" "${output_dir}"

        if [ ! -s "${output_dir}/${spk_embed_tag}.scp" ]; then
            log "Error: Speaker embedding extraction produced no utterance scp: ${output_dir}/${spk_embed_tag}.scp"
            exit 1
        fi
        # Parallel extraction completes batches out of order. ESPnet's
        # iterable dataset requires every input SCP to have the same key order.
        LC_ALL=C sort -k1,1 "${output_dir}/${spk_embed_tag}.scp" \
            -o "${output_dir}/${spk_embed_tag}.scp"
    done
fi

if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]7[[:space:]] ]]; then
    if "${use_default_centroid}"; then
        log "Stage 7: Train ${nclusters}-cluster K-means on unperturbed ${train_set} using ${kmeans_feature}"
        if [ ! -f "${kmeans_script}" ]; then
            log "Error: K-means utility was not found: ${kmeans_script}"
            exit 1
        fi
        if [ ! -f "${data_feats}/${train_set}/wav.scp" ]; then
            log "Error: Missing unperturbed stage 4 output: ${data_feats}/${train_set}/wav.scp"
            exit 1
        fi

        # Only dump the training features and learn K-means. Pseudo-labels are
        # unnecessary because the differentiable quantizer reads this model.
        # shellcheck disable=SC2086
        "${kmeans_script}" \
            --stage 1 \
            --stop_stage 2 \
            --train_set "${train_set}" \
            --dev_set "" \
            --other_sets "" \
            --datadir "${data_feats}" \
            --featdir "${dumpdir}/kmeans_features" \
            --audio_format "${audio_format}" \
            --audio_sample_rate 16000 \
            --feature_type "${kmeans_feature_type}" \
            --layer "${kmeans_layer}" \
            --feature_conf "${kmeans_feature_conf}" \
            --km_dir "${kmeans_dir}" \
            --portion "${kmeans_portion}" \
            --nclusters "${nclusters}" \
            --storage_save_mode "${kmeans_storage_save_mode}" \
            --use_gpu "${kmeans_use_gpu}" \
            --num_threads "${kmeans_num_threads}" \
            --nj "${nj}" \
            --python "${python}" \
            --cpu_cmd "${train_cmd}" \
            --cuda_cmd "${cuda_cmd}" \
            ${kmeans_opts}

        if [ ! -f "${resolved_centroid_path}" ]; then
            log "Error: K-means training did not produce ${resolved_centroid_path}"
            exit 1
        fi
        log "K-means centroid model: ${resolved_centroid_path}"
    else
        log "Skip stage 7: Centroid path will be read from the training config"
    fi
fi

if [ "${stage}" -le 8 ] && [ "${stop_stage}" -ge 8 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]8[[:space:]] ]]; then
    train_dir=${data_feats}/${effective_train_set}
    valid_dir=${data_feats}/${valid_set}
    train_spemb=${dumpdir}/${spk_embed_tag}/${effective_train_set}/${spk_embed_tag}.scp
    valid_spemb=${dumpdir}/${spk_embed_tag}/${valid_set}/${spk_embed_tag}.scp
    log "Stage 8: Collect input shapes for ${tok_task}"

    required_stats_files=(
        "${train_dir}/wav.scp" "${train_dir}/text"
        "${valid_dir}/wav.scp" "${valid_dir}/text"
        "${train_spemb}" "${valid_spemb}"
    )
    for required_file in "${required_stats_files[@]}"; do
        if [ ! -s "${required_file}" ]; then
            log "Error: Required collect-stats input is missing: ${required_file}"
            exit 1
        fi
    done
    stats_logdir=${tok_stats_dir}/logdir
    mkdir -p "${stats_logdir}"
    stats_nj=$(min "${nj}" "$(wc -l < "${train_dir}/wav.scp")" "$(wc -l < "${valid_dir}/wav.scp")")
    train_splits=()
    valid_splits=()
    for job in $(seq "${stats_nj}"); do
        train_splits+=("${stats_logdir}/train.${job}.scp")
        valid_splits+=("${stats_logdir}/valid.${job}.scp")
    done
    utils/split_scp.pl "${train_dir}/wav.scp" "${train_splits[@]}"
    utils/split_scp.pl "${valid_dir}/wav.scp" "${valid_splits[@]}"

    stats_config_opts=()
    if [ -n "${tok_config}" ]; then
        stats_config_opts+=(--config "${tok_config}")
    fi
    if "${use_default_centroid}"; then
        # collect_stats still constructs the model, so the centroid required
        # by the default config must be supplied here as it is for stage 9.
        stats_config_opts+=(--quantizer_conf "centroid_path=${resolved_centroid_path}")
    fi
    ${train_cmd} JOB=1:"${stats_nj}" "${stats_logdir}/stats.JOB.log" \
        "${python}" -m "espnet2.bin.${tok_task}_train" \
            --collect_stats true \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --bpemodel "${bpemodel}" \
            --train_data_path_and_name_and_type "${train_dir}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type "${train_dir}/text,text,text" \
            --valid_data_path_and_name_and_type "${valid_dir}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type "${valid_dir}/text,text,text" \
            --train_data_path_and_name_and_type "${train_spemb},spembs,kaldi_ark" \
            --valid_data_path_and_name_and_type "${valid_spemb},spembs,kaldi_ark" \
            --train_shape_file "${stats_logdir}/train.JOB.scp" \
            --valid_shape_file "${stats_logdir}/valid.JOB.scp" \
            --output_dir "${stats_logdir}/stats.JOB" \
            "${stats_config_opts[@]}" ${tok_args}

    aggregate_opts=()
    for job in $(seq "${stats_nj}"); do
        aggregate_opts+=(--input_dir "${stats_logdir}/stats.${job}")
    done
    "${python}" -m espnet2.bin.aggregate_stats_dirs \
        "${aggregate_opts[@]}" --output_dir "${tok_stats_dir}"
    for split in train valid; do
        awk -v vocab_size="$(wc -l < "${token_list}")" \
            '{print $0 "," vocab_size}' \
            "${tok_stats_dir}/${split}/text_shape" \
            > "${tok_stats_dir}/${split}/text_shape.${token_type}"
    done
fi

if [ "${stage}" -le 9 ] && [ "${stop_stage}" -ge 9 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]9[[:space:]] ]]; then
    train_dir=${data_feats}/${effective_train_set}
    valid_dir=${data_feats}/${valid_set}
    train_spemb=${dumpdir}/${spk_embed_tag}/${effective_train_set}/${spk_embed_tag}.scp
    valid_spemb=${dumpdir}/${spk_embed_tag}/${valid_set}/${spk_embed_tag}.scp
    log "Stage 9: Train speech tokenizer: ${tok_exp}"
    mkdir -p "${tok_exp}"

    train_config_opts=()
    if [ -n "${tok_config}" ]; then
        train_config_opts+=(--config "${tok_config}")
    fi
    quantizer_opts=()
    if "${use_default_centroid}"; then
        quantizer_opts+=(--quantizer_conf "centroid_path=${resolved_centroid_path}")
    fi
    jobname=${tok_exp}/train.log
    if echo "${cuda_cmd}" | grep -q -e queue.pl -e queue-freegpu.pl; then
        jobname=$(basename "${tok_exp}")
    fi
    "${python}" -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${tok_exp}/train.log" \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${tok_exp}/.dist_init_" \
        --multiprocessing_distributed true -- \
        "${python}" -m "espnet2.bin.${tok_task}_train" \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --bpemodel "${bpemodel}" \
            --resume true \
            "${quantizer_opts[@]}" \
            --train_data_path_and_name_and_type "${train_dir}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type "${train_dir}/text,text,text" \
            --valid_data_path_and_name_and_type "${valid_dir}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type "${valid_dir}/text,text,text" \
            --train_data_path_and_name_and_type "${train_spemb},spembs,kaldi_ark" \
            --valid_data_path_and_name_and_type "${valid_spemb},spembs,kaldi_ark" \
            --train_shape_file "${tok_stats_dir}/train/speech_shape" \
            --train_shape_file "${tok_stats_dir}/train/text_shape.${token_type}" \
            --valid_shape_file "${tok_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${tok_stats_dir}/valid/text_shape.${token_type}" \
            --train_shape_file "${tok_stats_dir}/train/spembs_shape" \
            --valid_shape_file "${tok_stats_dir}/valid/spembs_shape" \
            --output_dir "${tok_exp}" \
            "${train_config_opts[@]}" ${tok_args}
fi

if [ "${stage}" -le 10 ] && [ "${stop_stage}" -ge 10 ] \
        && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    log "Stage 10: Extract discrete tokens only"
    if "${gpu_inference}"; then
        inference_cmd=${cuda_cmd}
        inference_ngpu=1
    else
        inference_cmd=${decode_cmd}
        inference_ngpu=0
    fi
    for dset in ${test_sets}; do
        decode_data=${data_feats}/${dset}
        decode_dir=${tok_exp}/${inference_tag}/${dset}
        decode_logdir=${decode_dir}/logdir
        mkdir -p "${decode_logdir}"
        decode_nj=$(min "${inference_nj}" "$(wc -l < "${decode_data}/wav.scp")")
        decode_splits=()
        for job in $(seq "${decode_nj}"); do
            decode_splits+=("${decode_logdir}/keys.${job}.scp")
        done
        utils/split_scp.pl "${decode_data}/wav.scp" "${decode_splits[@]}"

        ${inference_cmd} --gpu "${inference_ngpu}" JOB=1:"${decode_nj}" \
            "${decode_logdir}/tokenize.JOB.log" \
            "${python}" -m espnet2.bin.tok_inference \
                --ngpu "${inference_ngpu}" \
                --batch_size "${inference_batch_size}" \
                --data_path_and_name_and_type "${decode_data}/wav.scp,speech,sound" \
                --key_file "${decode_logdir}/keys.JOB.scp" \
                --train_config "${tok_exp}/config.yaml" \
                --model_file "${tok_exp}/${inference_model}" \
                --output_dir "${decode_logdir}/output.JOB"

        for output_name in token token_length; do
            for job in $(seq "${decode_nj}"); do
                cat "${decode_logdir}/output.${job}/${output_name}"
            done | sort -k1 > "${decode_dir}/${output_name}"
        done
    done
fi

log "Finished stages ${stage}-${stop_stage}. Elapsed time: ${SECONDS}s"
