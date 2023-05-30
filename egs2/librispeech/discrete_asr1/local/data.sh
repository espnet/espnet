#!/usr/bin/env bash

# Copyright 2023 Xuankai Chang (Carnegie Mellon University)
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
SECONDS=0


stage=1
stop_stage=100000
skip_stages=         # Spicify the stage to be skipped
nj=16
gpu_nj=4
python=python3       # Specify python to execute espnet commands.
feats_root=data/extracted_features
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.

train_set="train_960"
dev_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

src_lang="dc"
src_case="rm"
tgt_lang="en"
tgt_case="ts"

# Data related
# typically this indicates:
#   ssl_model_type (hubert / wavlm) + 
#   ssl_model_version (base / large) + 
#   layer_idx +
feats_type="wavlm+large+24"
# Kmeans related
skip_train_kmeans=false
portion=0.1
nclusters=2000
# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

measure_label_quality=true


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

model_type=$(echo "${feats_type}" | cut -d+ -f1)
model_version=$(echo "${feats_type}" | cut -d+ -f2)
layer=$(echo "${feats_type}" | cut -d+ -f3)


if "${skip_train_kmeans}"; then
    skip_stages+="4 "
fi
if [ -z "${speed_perturb_factors}" ]; then
    skip_stages+="5 6"
fi
if ! "${measure_label_quality}"; then
    skip_stages+="8 9 "
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "stage 1: Preparing train / dev / test sets following ASR recipe."
    local/data_asr.sh --stage 1 --stop-stage 4
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    log "stage 2: Dumping the audio files."
    for dset in ${test_sets} "${train_set}" ; do
        utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" data/dump_audio/"${dset}"
        rm -f data/dump_audio/"${dset}"/{segments,wav.scp,reco2file_and_channel,reco2dur}

        _opts=
        if [ -e "data/${dset}"/segments ]; then
            # "segments" is used for splitting wav files which are written in "wav".scp
            # into utterances. The file format of segments:
            #   <segment_id> <record_id> <start_time> <end_time>
            #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
            # Where the time is written in seconds.
            _opts+="--segments data/${dset}/segments "
        fi
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
            "data/${dset}/wav.scp" "data/dump_audio/${dset}"
    done
fi

function download_model_ckpts() {
    log "download pretrained SSL models"
    ckpt_type=$1
    ckpt_version=$2

    if [ ${ckpt_type} == "hubert" ]; then
        # HuBERT
        if [ ${ckpt_version} == "base" ]; then
            # HuBERT_base
            model_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
        elif [ ${ckpt_version} == "large" ]; then
            # HuBERT_large
            model_url="https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt"
        else
            echo "Unsupported hubert version ${ckpt_version}" && exit 1;
        fi
        # shellcheck disable=SC2206
        ckpt_name=$(echo ${model_url} | awk -F/ '{print $NF}')
    elif [ ${ckpt_type} == "wavlm" ]; then
        # WavLM
        if [ ! -d "local/wavlm" ]; then
            echo "Download wavlm model packages"
            mkdir -p "local/wavlm"
            wget -O "local/wavlm/WavLM.py" "https://raw.githubusercontent.com/microsoft/unilm/master/wavlm/WavLM.py"
            wget -O "local/wavlm/modules.py" "https://raw.githubusercontent.com/microsoft/unilm/master/wavlm/modules.py"
        else
            echo "local/wavlm already exists. Skip downloading"
        fi

        if [ ${ckpt_version} == "base" ]; then
            # WavLM_base
            model_url="https://valle.blob.core.windows.net/share/wavlm/WavLM-Base.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"
        elif [ ${ckpt_version} == "large" ]; then
            # WavLM_large
            model_url="https://valle.blob.core.windows.net/share/wavlm/WavLM-Large.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"
        else
            echo "Unsupported wavlm version ${ckpt_version}" && exit 1;
        fi
        ckpt_name="wavlm_${ckpt_version}.pt"
    else
        echo "Unsupported ssl type ${ckpt_type}" && exit 1;
    fi
}

feats_dir=${feats_root}/${model_type}_${model_version}/l${layer}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    log "Stage 3: dump features"

    # download ssl ckpt
    mkdir -p "./ckpt"

    download_model_ckpts "${model_type}" "${model_version}"
    if [ ! -f "./ckpt/${ckpt_name}" ]; then
        wget -O "./ckpt/${ckpt_name}" ${model_url}
    fi

    for dset in ${test_sets} "${train_set}" ; do
        log "Dump SSL ${dset} features to ${feats_dir}/${dset}"

        # 1. Split the key file
        utils/copy_data_dir.sh --validate_opts --non-print data/dump_audio/"${dset}" "${feats_dir}/${dset}"
        rm -f "${feats_dir}/${dset}"/{segments,wav.scp,reco2file_and_channel,reco2dur}

        output_dir="${feats_dir}/${dset}/data"
        mkdir -p "${output_dir}"
        _logdir="${feats_dir}/${dset}/logdir"
        mkdir -p "${_logdir}"

        nutt=$(<"data/dump_audio/${dset}"/wav.scp wc -l)
        _nj=$((gpu_nj<nutt?gpu_nj:nutt))

        key_file="data/dump_audio/${dset}"/wav.scp
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/wav.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        ${cuda_cmd} --gpu 1 JOB=1:${_nj} ${_logdir}/dump_features.JOB.log \
            ${python} local/dump_hubert_or_wavlm_feature.py \
                --ckpt_path "./ckpt/${ckpt_name}" \
                --layer ${layer} \
                --ssl_type "${model_type}" \
                --in_filetype "sound" \
                --out_filetype "mat" \
                --write_num_frames "ark,t:${output_dir}/utt2num_frames.JOB" \
                "scp:${_logdir}/wav.JOB.scp" \
                "ark,scp:${output_dir}/feats.JOB.ark,${output_dir}/feats.JOB.scp" || exit 1;

        # concatenate scp files
        for n in $(seq ${_nj}); do
            cat ${output_dir}/feats.${n}.scp || exit 1;
        done > ${output_dir}/../feats.scp || exit 1

        for n in $(seq ${_nj}); do
            cat ${output_dir}/utt2num_frames.$n || exit 1;
        done > ${output_dir}/../utt2num_frames || exit 1

    done

fi

km_dir=exp/kmeans/$(echo "${feats_type}" | tr "+" "_")_${nclusters}clusters
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    log "Stage 4: learn kmeans"
    mkdir -p ${km_dir}

    # select portion of data
    nutt=$(<"${feats_dir}/${train_set}"/feats.scp wc -l)
    portion_nutt=$(echo ${nutt} ${portion} | awk '{print(int($1 * $2)+1)}')

    subset_scp.pl ${portion_nutt} "${feats_dir}/${train_set}"/feats.scp \
        > "${km_dir}/train.scp" || exit 1;
    log "Subsampling ${portion_nutt} utterances from ${train_set} for Kmeans training."

    # It typically requires 120GB RAM to run kmeans steps.
    ${train_cmd} --num_threads 12 --mem 15G ${km_dir}/logdir/learn_kmeans.log \
        ${python} local/learn_kmeans.py \
            --km_path ${km_dir}/km_${nclusters}.mdl \
            --n_clusters ${nclusters} \
            --percent "-1" \
            --in_filetype "mat" \
            "scp:${km_dir}/train.scp" || exit 1;
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    log "Stage 5: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
    for factor in ${speed_perturb_factors}; do
        if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
            scripts/utils/perturb_data_dir_speed.sh \
                "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
            _dirs+="data/${train_set}_sp${factor} "

            utils/copy_data_dir.sh --validate_opts --non-print data/"${train_set}_sp${factor}" data/dump_audio/"${train_set}_sp${factor}"
            rm -f data/dump_audio/"${train_set}_sp${factor}"/{segments,wav.scp,reco2file_and_channel,reco2dur}

            _opts=
            if [ -e "data/${train_set}_sp${factor}"/segments ]; then
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${train_set}_sp${factor}/segments "
            fi
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${train_set}_sp${factor}/wav.scp" "data/dump_audio/${train_set}_sp${factor}"
        else
            # If speed factor is 1, same as the original
            _dirs+="data/${train_set} "
        fi
    done
    utils/combine_data.sh "data/${train_set}_sp" ${_dirs}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && ! [[ " ${skip_stages} " =~ [[:space:]]6[[:space:]] ]]; then
    log "Stage 6: Dump SSL ${train_set}_sp* features to ${feats_dir}/${train_set}_sp*"

    download_model_ckpts "${model_type}" "${model_version}"

    _dirs=
    for factor in ${speed_perturb_factors}; do
        if ${python} -c "assert ${factor} != 1.0" 2>/dev/null; then
            utils/copy_data_dir.sh --validate_opts --non-print data/dump_audio/"${train_set}_sp${factor}" "${feats_dir}/${train_set}_sp${factor}"
            rm -f "${feats_dir}/${train_set}_sp${factor}"/{segments,wav.scp,reco2file_and_channel,reco2dur}

            _dirs+="${feats_dir}/${train_set}_sp${factor} "
            # 1. Split the key file
            output_dir="${feats_dir}/${train_set}_sp${factor}/data"
            mkdir -p "${output_dir}"
            _logdir="${feats_dir}/${train_set}_sp${factor}/logdir"
            mkdir -p "${_logdir}"
            nutt=$(<"data/dump_audio/${train_set}_sp${factor}"/wav.scp wc -l)
            _nj=$((gpu_nj<nutt?gpu_nj:nutt))

            key_file="data/dump_audio/${train_set}_sp${factor}"/wav.scp
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/wav.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            ${cuda_cmd} --gpu 1 JOB=1:${_nj} ${_logdir}/dump_features.JOB.log \
                ${python} local/dump_hubert_or_wavlm_feature.py \
                    --ckpt_path "./ckpt/${ckpt_name}" \
                    --layer ${layer} \
                    --ssl_type "${model_type}" \
                    --in_filetype "sound" \
                    --out_filetype "mat" \
                    --write_num_frames "ark,t:${output_dir}/utt2num_frames.JOB" \
                    "scp:${_logdir}/wav.JOB.scp" \
                    "ark,scp:${output_dir}/feats.JOB.ark,${output_dir}/feats.JOB.scp" || exit 1;

            # concatenate scp files
            for n in $(seq ${_nj}); do
                cat ${output_dir}/feats.${n}.scp || exit 1;
            done > ${output_dir}/../feats.scp || exit 1

            for n in $(seq ${_nj}); do
                cat ${output_dir}/utt2num_frames.$n || exit 1;
            done > ${output_dir}/../utt2num_frames || exit 1
        else
            # If speed factor is 1, same as the original
            _dirs+="${feats_root}/${model_type}_${model_version}/l${layer}/${train_set} "
        fi
    done

    utils/combine_data.sh \
        --extra_files "feats.scp utt2num_frames" \
        "${feats_root}/${model_type}_${model_version}/l${layer}/${train_set}_sp" ${_dirs}
fi

if [ -n "${speed_perturb_factors}" ]; then
    train_set="${train_set}_sp"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && ! [[ " ${skip_stages} " =~ [[:space:]]7[[:space:]] ]]; then
    log "Stage 7: generate kmeans labels"

    for dset in ${test_sets} "${train_set}" ; do
        log "Extract labels to ${feats_dir}/${dset}"

        _logdir="${feats_dir}/${dset}/logdir"
        mkdir -p ${_logdir}

        nutt=$(<"${feats_dir}/${dset}/"feats.scp wc -l)
        _nj=$((nj<nutt?nj:nutt))

        key_file="${feats_dir}/${dset}"/feats.scp
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/inference_kmeans.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        ${train_cmd} JOB=1:${_nj} ${_logdir}/inference_pseudo_labels_km${nclusters}.JOB.log \
            ${python} local/dump_km_label.py \
                --km_path ${km_dir}/km_${nclusters}.mdl \
                --in_filetype "mat" \
                --out_filetype "mat" \
                "scp:${_logdir}/inference_kmeans.JOB.scp" \
                "ark,t:${_logdir}/pseudo_labels_km${nclusters}.JOB.txt" || exit 1;

        # concatenate scp files
        for n in $(seq ${_nj}); do
            cat ${_logdir}/pseudo_labels_km${nclusters}.${n}.txt || exit 1;
        done | sed 's/ \[ \| \]//g' > "${_logdir}"/../pseudo_labels_km${nclusters}.txt || exit 1;
    done
fi

alignment_phoneme_dir="./data/mfa_phoneme_alignment"
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && ! [[ " ${skip_stages} " =~ [[:space:]]8[[:space:]] ]]; then

    if [ ! -f ${alignment_phoneme_dir}/dev-clean.tsv ] && [ ! -f ${alignment_phoneme_dir}/dev-other.tsv ]; then
        log "Stage 8: Downloading MFA from https://zenodo.org/record/2619474#.Y2F3ZewVDu0"
        mkdir -p ${alignment_phoneme_dir}

        wget \
            -O ${alignment_phoneme_dir}/librispeech_alignments.zip \
            https://zenodo.org/record/2619474/files/librispeech_alignments.zip?download=1

        unzip "${alignment_phoneme_dir}/librispeech_alignments.zip" -d "${alignment_phoneme_dir}"
        python local/dump_librispeech_alignment_from_textgrid.py \
            --alignment_root "${alignment_phoneme_dir}" \
            --dataset "dev-other" "dev-clean"
    else
        log "Stage 8: Librispeech phoneme alignments exists. Skipping ..."
    fi
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] && ! [[ " ${skip_stages} " =~ [[:space:]]9[[:space:]] ]]; then
    log "Stage 9: measure qualities of labels: purity"

    # (simpleoier): For LibriSpeech task only
    ${python} local/measure_teacher_quality.py \
        --lab_dir "${feats_dir}" \
        --lab_name "pseudo_labels_km${nclusters}.txt" \
        --lab_sets "dev_clean" "dev_other" \
        --phn_dir "${alignment_phoneme_dir}" \
        --phn_sets "dev-clean" "dev-other" \
        --pad_len 0 \
        --upsample 2 \
        --ref_lab_dir "" \
        --ref_lab_name "" | tee ${km_dir}/phoneme_pseudo_label_quality.txt

fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    log "Stage 10: Prepare token_list and convert number indices to CJK tokens"

    # Get uniq chars
    if [ ! -f "${km_dir}/../"distinct_cjk_token_lists ]; then
        ${python} -c "for i in range(4096): print(i, chr(int('4e00', 16) + i))" \
            > "${km_dir}/../"distinct_cjk_token_lists
    fi

    if [ "${src_case}" = ts ]; then
        echo "keep the original discrete token sequence"
        for dset in "${train_set}" ${test_sets}; do
            awk '
                (FILENAME==ARGV[1]) {a[$1]=$2}
                (FILENAME==ARGV[2]) {
                    out="";
                    for (i=2; i<=NF; i++) {
                        out=out""a[$i];
                    }
                    print($1,out);
                }' "${km_dir}/../"distinct_cjk_token_lists \
                ${feats_dir}/${dset}/pseudo_labels_km${nclusters}.txt \
                > "data/${dset}"/text.${src_case}.${src_lang}
        done
    elif [ "${src_case}" = rm ]; then
        echo "remove repetitions in the discrete token sequence"
        for dset in "${train_set}" ${test_sets}; do
            awk '
                (FILENAME==ARGV[1]) {a[$1]=$2}
                (FILENAME==ARGV[2]) {
                    out="";
                    for (i=2; i<=NF; i++) {
                        if ($i != $(i-1)) {out=out""a[$i]}
                    }
                    print($1,out);
                }' "${km_dir}/../"distinct_cjk_token_lists \
                ${feats_dir}/${dset}/pseudo_labels_km${nclusters}.txt \
                > "data/${dset}"/text.${src_case}.${src_lang}
        done
    else
        echo "Unrecognized src_case ${src_case}" && exit 1;
    fi
    # (simpleoier) For LibriSpeech only
    awk '(FILENAME == ARGV[1] || FILENAME == ARGV[2]) {a[$1]=$0} (FILENAME == ARGV[3]) {print(a[$1])}' \
        data/dev_clean/text.${src_case}.${src_lang} \
        data/dev_other/text.${src_case}.${src_lang} \
        data/${dev_set}/wav.scp > data/${dev_set}/text.${src_case}.${src_lang}

    for dset in "${train_set}" "${dev_set}" ${test_sets}; do
        cp data/${dset}/text data/${dset}/text.${tgt_case}.${tgt_lang}
    done

fi