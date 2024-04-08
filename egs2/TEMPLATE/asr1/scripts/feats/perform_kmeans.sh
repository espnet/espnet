#!/bin/bash
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
skip_stages=
cpu_cmd="run.pl"
num_threads=20      # number of cpu threads in learn_kmeans
cuda_cmd="run.pl"
nj=16               # number of parallel jobs
python=python3      # Specify python to execute espnet commands.
train_set=          # Name of training set
dev_set=            # Name of valid set
other_sets=         # Name of other sets
datadir=dump/raw    # Directory for the source speech data used to dump feature and label.
featdir=dump/hubert_feats   # Directory for the dumped features and labels.
km_dir=             # Directory for the kmeans models
dictdir=            # Directory for the fairseq dictionary (only used for hubert training)
alignment_phoneme_dir="data/mfa_phoneme_alignment"  # Directory for alignment labels
phn_sets="dev-other dev-clean"      # Datasets of alignment used to measure the pseudo-label quality
upsample=           # Upsampling rate of pseudo-labels to measure the pseudo-lable quality
use_gpu=false       # Whether to use gpu in feature extraction
suffix=             # A suffix to distinguish the feature dump directory. Empty in usual cases.
audio_format="wav"  # The audio format of the source speech (flac, wav, *_ark, etc)

skip_train_kmeans=false     # Whether to skip the kmeans model training
nclusters=100       # Number of clusters of kmeans model
portion=0.1         # Portion of data from training set used to train kmeans model
storage_save_mode=false     # Save storage on SSL feature extraction
                            # If true, feature extraction and kmeans clustering on the fly

feature_conf=       # feature configuration in json string format
feature_type=mfcc   # mfcc / fairseq_hubert / espnet_hubert
layer=              # The layer index of SSL models to extract features from.
batch_bins=         # batch size when extracting features and labels.

# Legacy Fairseq HuBERT model and ESPnet-trained HuBERT model related for feature extraction.
# Example of legacy Fairseq HuBERT model
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models/hubert_base_ls960.pt"
# Example of espnet-trained model
# hubert_url="espnet"
# hubert_dir_path="" # Pretrained Hubert model dir contains 'valid.acc.best.pth' and 'config.yaml'

log "$0 $*"
. utils/parse_options.sh

. ./path.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 <--nclusters:100> <--feature_type:mfcc>"
    exit 0
fi

if [ ${feature_type} = "mfcc" ]; then  # MFCC has no layer
    use_gpu=false
elif [ -z "${suffix}" ]; then
    suffix="layer${layer}/"
fi
if [ -z "${feature_conf}" ]; then
    feature_conf="{type=${feature_type}"
    if [ ${feature_type} = "espnet_hubert" ]; then
        feature_conf+=",conf={\
sample_rate=16000,hubert_model_path=${hubert_dir_path},\
layer=${layer}\
}"
    elif [ ${feature_type} = "fairseq_hubert" ]; then
        feature_conf+=",conf={\
sample_rate=16000,hubert_url=${hubert_url},\
hubert_dir_path=${hubert_dir_path},layer=${layer}\
}"
    elif [ ${feature_type} != "mfcc" ]; then
        log "Error: unsupported feature type ${feature_type}" && exit 2
    fi
    feature_conf+="}"
fi

if "${skip_train_kmeans}"; then
    skip_stages+=" 2"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "stage 1: Dump ${feature_type} feature"

    if ${use_gpu}; then
        _cmd="${cuda_cmd} --gpu 1"
    else
        _cmd="${cpu_cmd}"
    fi

    if [[ "${audio_format}" == *ark* ]]; then
        _in_filetype="kaldi_ark"
    else
        # "sound" supports "wav", "flac", etc.
        _in_filetype="sound"
    fi

    if ${storage_save_mode}; then
        _dsets="${train_set}_subset${portion}"
        mkdir -p "${datadir}/${_dsets}"

        nutt=$(<"${datadir}/${train_set}"/wav.scp wc -l)
        portion_nutt=$(echo ${nutt} ${portion} | awk '{print(int($1 * $2))}')
	portion_nutt=$(( portion_nutt > 0 ? portion_nutt : 1 ))

        utils/subset_data_dir.sh \
            "${datadir}/${train_set}" ${portion_nutt} "${datadir}/${_dsets}"
        utils/filter_scp.pl ${datadir}/${_dsets}/utt2spk \
            <${datadir}/${train_set}/utt2num_samples >${datadir}/${_dsets}/utt2num_samples
        log "Subsampling ${portion_nutt} utterances for feature dumping."
    else
        _dsets="${train_set} ${other_sets} ${dev_set}"
    fi
    for dset in ${_dsets}; do
        echo "Dump SSL ${dset} features to ${featdir}/${feature_type}/${suffix}${dset}"
        _dump_dir="${featdir}/${feature_type}/${suffix}${dset}"

        utils/copy_data_dir.sh --validate_opts --non-print "${datadir}/${dset}" "${_dump_dir}"

        # 1. Split the key file
        output_dir="${_dump_dir}/data"
        mkdir -p "${output_dir}"
        _logdir="${_dump_dir}/logdir"
        mkdir -p "${_logdir}"

        nutt=$(<"${_dump_dir}"/wav.scp wc -l)
        _nj=$((nj<nutt?nj:nutt))

        key_file="${datadir}/${dset}"/wav.scp
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/wav.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        for n in $(seq ${_nj}); do
            awk '(FILENAME==ARGV[1]){utt2num[$1]=$2} (FILENAME==ARGV[2]){print($1, utt2num[$1])}' \
                ${datadir}/${dset}/utt2num_samples ${_logdir}/wav.${n}.scp > ${_logdir}/utt2num_samples.${n}
        done

        # shellcheck disable=SC2046,SC2086
        ${_cmd} JOB=1:${_nj} ${_logdir}/dump_features.JOB.log \
            ${python} pyscripts/feats/dump_ssl_feature.py \
                --feature_conf "'${feature_conf}'" \
                --use_gpu ${use_gpu} \
                --in_filetype "${_in_filetype}" \
                --out_filetype "mat" \
                --write_num_frames "ark,t:${output_dir}/utt2num_frames.JOB" \
                --utt2num_samples "${_logdir}/utt2num_samples.JOB" \
                ${batch_bins:+--batch_bins ${batch_bins}} \
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


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    log "stage 2: Learn K-means with ${feature_type} feature based on scikit-learn"

    _logdir="${km_dir}/logdir"
    mkdir -p ${_logdir}

    if ${storage_save_mode}; then
        _portion=1.0
        _dset="${train_set}_subset${portion}"
    else
        _portion=${portion}
        _dset="${train_set}"
    fi

    # select portion of data
    if (( $(echo "${_portion} >= 1.0" | bc -l) )); then
        cp "${featdir}/${feature_type}/${suffix}${_dset}"/feats.scp "${km_dir}/train.scp"
    else
        nutt=$(<"${featdir}/${feature_type}/${suffix}${_dset}"/feats.scp wc -l)
        portion_nutt=$(echo ${nutt} ${_portion} | awk '{print(int($1 * $2)+1)}')

        subset_scp.pl \
            ${portion_nutt} ${featdir}/${feature_type}/${suffix}${_dset}/feats.scp \
            > "${km_dir}/train.scp" || exit 1;
        log "Subsampling ${portion_nutt} utterances for Kmeans training."
    fi

    # It typically requires 120GB RAM to run kmeans steps.
    ${cpu_cmd} --num_threads ${num_threads} ${_logdir}/learn_kmeans.log \
        ${python} pyscripts/utils/learn_kmeans.py \
            --km_path ${km_dir}/km_${nclusters}.mdl \
            --n_clusters ${nclusters} \
            --percent -1 \
            --in_filetype mat \
            "scp:${km_dir}/train.scp" || exit 1;
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    log "stage 3: Generate K-means pseudo-labels"

    if ${use_gpu}; then
        _cmd="${cuda_cmd} --gpu 1"
    else
        _cmd="${cpu_cmd}"
    fi

    for dset in "${train_set}" "${dev_set}" ${other_sets}; do
        log "Extract labels to ${featdir}/${feature_type}/${suffix}${dset}"

        _dump_dir="${featdir}/${feature_type}/${suffix}${dset}"

        _opts=
        if ${storage_save_mode}; then
            utils/copy_data_dir.sh --validate_opts --non-print "${datadir}/${dset}" "${_dump_dir}"
            key="wav.scp"
            if [[ "${audio_format}" == *ark* ]]; then
                _opts+="--in_filetype kaldi_ark "
            else
                # "sound" supports "wav", "flac", etc.
                _opts+="--in_filetype sound "
            fi
            _opts+="--online_feature_extract ${storage_save_mode} "
            _opts+="--feature_conf \"${feature_conf}\" "
            if [ -n "${batch_bins}" ]; then
                _opts+="--batch_bins ${batch_bins} "
            fi
        else
            key="feats.scp"
            _opts+="--in_filetype mat "
        fi
        mkdir -p "${_dump_dir}"/logdir

        nutt=$(<"${_dump_dir}"/${key} wc -l)
        _nj=$((nj<nutt?nj:nutt))

        key_file="${_dump_dir}"/${key}
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_dump_dir}/logdir/inference_kmeans.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        for n in $(seq ${_nj}); do
            awk '(FILENAME==ARGV[1]){utt2num[$1]=$2} (FILENAME==ARGV[2]){print($1, utt2num[$1])}' \
                ${datadir}/${dset}/utt2num_samples ${_dump_dir}/logdir/inference_kmeans.${n}.scp \
                > ${_dump_dir}/logdir/utt2num_samples.${n}
        done

        ${_cmd} JOB=1:${_nj} "${_dump_dir}"/logdir/inference_pseudo_labels_km${nclusters}.JOB.log \
            ${python} pyscripts/feats/dump_km_label.py \
                ${_opts} \
                --km_path "${km_dir}/km_${nclusters}.mdl" \
                --out_filetype "mat" \
                --use_gpu ${use_gpu} \
                --utt2num_samples "${_dump_dir}/logdir/utt2num_samples.JOB" \
                "scp:${_dump_dir}/logdir/inference_kmeans.JOB.scp" \
                "ark,t:${_dump_dir}/logdir/pseudo_labels_km${nclusters}.JOB.txt" || exit 1;

        # concatenate scp files
        for n in $(seq ${_nj}); do
            cat "${_dump_dir}"/logdir/pseudo_labels_km${nclusters}.${n}.txt || exit 1;
        done | sed 's/ \[ \| \]//g' | sort -u > "${_dump_dir}"/pseudo_labels_km${nclusters}.txt || exit 1;
    done
fi


km_tag=$(basename ${km_dir})

if [ -n "${alignment_phoneme_dir}" ]; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
        log "Stage 4: Measure qualities of pseudo labels"

        if [ -z "${upsample}" ]; then
            # upsample the pseudo labels to match the length of alignment
            if [ "${feature_type}" = "mfcc" ]; then
                upsample=1
            else
                upsample=2
            fi
        fi

        if [ -d "${alignment_phoneme_dir}" ]; then
            # TODO(simpleoier): This script and arguments design are specific to LibriSpeech dataset.
            ${python} local/measure_teacher_quality.py \
                --lab_dir "${featdir}/${feature_type}/${suffix}" \
                --lab_name "pseudo_labels_km${nclusters}.txt" \
                --lab_sets "${dev_set}" \
                --phn_dir "${alignment_phoneme_dir}" \
                --phn_sets ${phn_sets} \
                --pad_len 0 \
                --upsample ${upsample} \
                --ref_lab_dir "" \
                --ref_lab_name "" | tee ${km_dir}/phoneme_pseudo_label_quality.txt
        else
            log "Skipping quality measurement because no ${alignment_phoneme_dir} exists. You can specify the \
alignment by \"--alignment_phoneme_dir\". The alignment is in tsv file with format: \"utt_id1 a1,a2,a3,...\""
        fi
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    log "stage 5: Prepare pseudo-labels for training and dictionary: <token> <count>"

    for dset in "${train_set}" "${dev_set}" ${other_sets}; do
        label_dir="${featdir}/${feature_type}/${suffix}${dset}"
        if [ -f "${label_dir}"/pseudo_labels_km${nclusters}.txt ]; then
            cp "${label_dir}"/pseudo_labels_km${nclusters}.txt ${datadir}/${dset}/text.km.${km_tag}
        fi
        utils/fix_data_dir.sh --utt_extra_files "text.km.${km_tag}" ${datadir}/${dset}
    done

    # generate dictionaries
    if [ -n "${dictdir}" ]; then
        mkdir -p ${dictdir}

        oov="<unk>"         # Out of vocabulary symbol.
        blank="<blank>"     # CTC blank symbol
        pad="<pad>"
        sos_eos="<sos/eos>" # sos and eos symbole

        <${datadir}/${train_set}/text.km.${km_tag} cut -d" " -f2- | \
            awk '{for (i=1; i<=NF; i++) {count[$i]+=1}} END{for (k in count) {print(k, count[k])}}' | \
                sort -n -r -k 2  | \
                awk -v oov=${oov} -v blank=${blank} -v sos_eos=${sos_eos} -v pad=${pad} \
                    '{print($1)} END{print(oov); print(sos_eos)}' \
                > ${dictdir}/tokens.txt

        log "Successfully generate the ${dictdir}/{dict,tokens}.txt"
    fi

fi
