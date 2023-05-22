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
python=python3       # Specify python to execute espnet commands.
train_set=
dev_set=
datadir=dump/raw
feat_dir=dump/hubert_feats
km_dir=
dictdir=
alignment_phoneme_dir=
phn_sets="dev-other dev-clean"
use_gpu=false

nclusters=100
feature_type=mfcc
layer=

# Extract intermediate Hubert embedding from official hubert model:
hubert_type="espnet"  # fairseq or espnet
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models/hubert_base_ls960.pt"

# Extract intermediate Hubert embedding from espnet-trained model:
# hubert_url="espnet"
# hubert_dir_path="" # Pretrained Hubert model dir contains 'valid.acc.best.pth' and 'config.yaml'

portion=0.1
nj=16
python=python3       # Specify python to execute espnet commands.

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 <--nclusters:100> <--feature_type:mfcc>"
    exit 0
fi


km_tag=$(basename ${km_dir})

if [ "${feature_type}" = "hubert" ]; then
    suffix="layer${layer}/"
else
    suffix=""
    use_gpu=false  # mfcc feature does not require GPU.
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dump ${feature_type} feature"

    if ${use_gpu}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${train_cmd}"
        _ngpu=0
    fi

    for dset in "${train_set}" "${dev_set}"; do
        echo "${dset}"

        # 1. Split the key file
        output_dir="${feat_dir}/${feature_type}/${suffix}${dset}/data"
        mkdir -p "${output_dir}"
        _logdir="${feat_dir}/${feature_type}/${suffix}${dset}/logdir"
        mkdir -p "${_logdir}"
        nutt=$(<"${datadir}/${dset}"/wav.scp wc -l)
        _nj=$((nj<nutt?nj:nutt))

        key_file="${datadir}/${dset}"/wav.scp
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/wav.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # shellcheck disableSC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/dump_feats.JOB.log \
            ${python} local/dump_mfcc_or_hubert_features.py \
                --in_filetype "sound" \
                --out_filetype "mat" \
                --feature_type "${feature_type}" \
                --hubert_type "${hubert_type}" \
                --hubert-model-url "${hubert_url}" \
                --hubert-model-path "${hubert_dir_path}" \
                --layer "${layer}" \
                --write_num_frames "ark,t:${_logdir}/utt2num_frames.JOB" \
                "scp:${_logdir}/wav.JOB.scp" \
                "ark,scp:${output_dir}/feats.JOB.ark,${output_dir}/feats.JOB.scp" || exit 1;

        # concatenate scp files
        for n in $(seq ${_nj}); do
            cat ${output_dir}/feats.${n}.scp || exit 1;
        done > ${output_dir}/../feats.scp || exit 1

        for n in $(seq ${_nj}); do
            cat ${_logdir}/utt2num_frames.$n || exit 1;
        done > ${output_dir}/../utt2num_frames || exit 1
        rm ${_logdir}/utt2num_frames.*

    done

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Learn K-means with ${feature_type} feature based on scikit-learn"

    _logdir="${km_dir}/logdir"
    mkdir -p ${_logdir}

    # select portion of data
    nutt=$(<"${feat_dir}/${feature_type}/${suffix}${train_set}"/feats.scp wc -l)
    portion_nutt=$(echo ${nutt} ${portion} | awk '{print(int($1 * $2 + 0.9))}')  # get ceil value
    subset_scp.pl \
        ${portion_nutt} ${feat_dir}/${feature_type}/${suffix}${train_set}/feats.scp \
        > "${km_dir}/train.scp" || exit 1;
    log "Subsampling ${portion_nutt} utterances for Kmeans training."

    ${train_cmd} ${_logdir}/learn_kmeans.log \
        ${python} local/learn_kmeans.py \
            --in_filetype mat \
            --km_path ${km_dir}/km_${nclusters}.mdl \
            --n_clusters ${nclusters} \
            --percent -1 \
            "scp:${km_dir}/train.scp" || exit 1;

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Generate K-means pseudo-labels"

    for dset in ${train_set} ${dev_set}; do
        echo ${dset}
        label_dir="${feat_dir}/${feature_type}/${suffix}${dset}/pseudo_labels"

        nutt=$(<"${feat_dir}/${feature_type}/${suffix}${dset}/"feats.scp wc -l)
        _nj=$((nj<nutt?nj:nutt))

        ${train_cmd} JOB=1:${_nj} ${label_dir}/logdir/dump_km_label.JOB.log \
            ${python} local/dump_km_label.py \
                --km_path "${km_dir}/km_${nclusters}.mdl" \
                --in_filetype "mat" \
                --out_filetype "mat" \
                "scp:${feat_dir}/${feature_type}/${suffix}${dset}/data/feats.JOB.scp" \
                "ark,t:${label_dir}/logdir/pseudo_labels.JOB.txt" || exit 1;

        # concatenate scp files
        for n in $(seq ${_nj}); do
            cat ${label_dir}/logdir/pseudo_labels.${n}.txt || exit 1;
        done | sed 's/ \[ \| \]//g' > "${label_dir}"/pseudo_labels.txt || exit 1;

        cp "${label_dir}"/pseudo_labels.txt ${datadir}/${dset}/text.km.${km_tag}

        utils/fix_data_dir.sh --utt_extra_files "text.km.${km_tag}" ${datadir}/${dset}
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Generate char-based fairseq style dictionary: <token> <count>"
    # generate dictionaries
    oov="<unk>"         # Out of vocabulary symbol.
    blank="<blank>"     # CTC blank symbol
    pad="<pad>"
    sos_eos="<sos/eos>" # sos and eos symbole

    mkdir -p ${dictdir}

    <${datadir}/${dset}/text.km.${km_tag} cut -d" " -f2- | \
        awk '{for (i=1; i<=NF; i++) {count[$i]+=1}} END{for (k in count) {print(k, count[k])}}' | \
            sort -n -r -k 2  | \
            awk -v oov=${oov} -v blank=${blank} -v sos_eos=${sos_eos} -v pad=${pad} \
                '{print($1)} END{print(oov); print(sos_eos)}' \
            > ${dictdir}/tokens.txt

    log "Successfully generate the ${dictdir}/{dict,tokens}.txt"

fi

if [ -n "${alignment_phoneme_dir}" ]; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Measure qualities of pseudo labels"

        if [ "${feature_type}" = "hubert" ]; then
            upsample=2
        else
            upsample=1
        fi

        ${python} local/measure_teacher_quality.py \
            --lab_dir ${datadir} \
            --lab_name "text.km.${km_tag}" \
            --lab_sets "${dev_set}" \
            --phn_dir "${alignment_phoneme_dir}" \
            --phn_sets ${phn_sets} \
            --pad_len 0 \
            --upsample ${upsample} \
            --ref_lab_dir "" \
            --ref_lab_name "" | tee ${km_dir}/phoneme_pseudo_label_quality.txt

    fi
fi
