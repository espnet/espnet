#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#               2022 Dongji Gao
#               2022 Carnegie Mellon University (Author: Jiatong Shi)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Begin configuration section.
set -e 
set -u 
set -o pipefail

cmd="run.pl"
nj=1
dim=512
num_clusters=128
train_set="train"
valid_set="valid"
test_sets=""
skip_training=false
reduce=false

# End configuration section.
echo "$0 $*" 1>&2 # Print the command line for logging

. utils/parse_options.sh || exit 1;

if ${skip_training}; then
    all_sets="${valid_set} ${test_sets}"
else
    all_sets="${train_set} ${valid_set} ${test_sets}"
fi

uasr_stats_dir=$1
output_feats_dir=$2

logdir="${uasr_stats_dir}/logdir"
echo "using ${dim} dim for PCA"

[ ! -f ${output_feats_dir} ] && mkdir -p "${output_feats_dir}"

train_feats_scp="${uasr_stats_dir}/${train_set}/collect_feats/feats.scp"
if "${reduce}"; then
    echo "$0: Reducing ${train_feats_scp}"
    sort "${train_feats_scp}" | awk 'NR % 10 == 0'  > ${uasr_stats_dir}/${train_set}/collect_feats/feats_reduced.scp
    train_feats_scp="${uasr_stats_dir}/${train_set}/collect_feats/feats_reduced.scp"
fi    

if ! ${skip_training}; then
    feats_scp="${uasr_stats_dir}/${train_set}/collect_feats/feats.scp"
    split_dir="${uasr_stats_dir}/${train_set}/collect_feats/split${nj}"
    mkdir -p "${split_dir}"
    train_split_feats_scp=""
    for n in $(seq ${nj}); do
        mkdir -p "${split_dir}/${n}"
        train_split_feats_scp="${train_split_feats_scp} ${split_dir}/${n}/feats.scp"        
    done
    utils/split_scp.pl "${feats_scp}" ${train_split_feats_scp}
fi

if [ -n "${valid_set}" ]; then
    feats_scp="${uasr_stats_dir}/${valid_set}/collect_feats/feats.scp"
    split_dir="${uasr_stats_dir}/${valid_set}/collect_feats/split${nj}"
    mkdir -p "${split_dir}"
    valid_split_feats_scp=""
    for n in $(seq ${nj}); do
        mkdir -p "${split_dir}/${n}"
        valid_split_feats_scp="${valid_split_feats_scp} ${split_dir}/${n}/feats.scp"        
    done
    utils/split_scp.pl "${feats_scp}" ${valid_split_feats_scp}
fi

for test_set in ${test_sets}; do
    if [ -n "${test_set}" ]; then
        feats_scp="${uasr_stats_dir}/${test_set}/collect_feats/feats.scp"
        split_dir="${uasr_stats_dir}/${test_set}/collect_feats/split${nj}"
        mkdir -p "${split_dir}"
        test_split_feats_scp=""
        for n in $(seq ${nj}); do
            mkdir -p "${split_dir}/${n}"
            test_split_feats_scp="${test_split_feats_scp} ${split_dir}/${n}/feats.scp"        
        done
        utils/split_scp.pl "${feats_scp}" ${test_split_feats_scp}
    fi
done

if ! ${skip_training}; then
    echo "Generating ${num_clusters} clusters"
    ${cmd} ${logdir}/generate_feats_cluster.log \
        python pyscripts/feats/feats_cluster_faiss.py \
            "${train_feats_scp}" \
            --save_dir "${output_feats_dir}" \
            -f "CLUS${num_clusters}" \
            --sample_pct 1.0

    echo "Computing PCA"
    ${cmd} ${logdir}/compute_pca.log \
        python pyscripts/feats/pca.py \
            "${train_feats_scp}" \
            --output "${output_feats_dir}/pca" \
            --dim $dim
fi

for split in ${all_sets}; do
    echo "Applying cluster on ${split}"
    ${cmd} JOB=1:${nj} ${logdir}/apply_cluster_${split}.JOB.log \
          python pyscripts/feats/feats_apply_cluster_faiss.py \
              "${uasr_stats_dir}/${split}/collect_feats/split${nj}/JOB/feats.scp" \
              --split "${split}" \
              --model_path ${output_feats_dir}/CLUS${num_clusters} \
              --output_path ${output_feats_dir}/CLUS${num_clusters}/JOB/ \
              -f "CLUS${num_clusters}"

    echo "Applying PCA on ${split}"
    ${cmd} JOB=1:${nj} ${logdir}/apply_pca_${split}.JOB.log \
        python pyscripts/feats/apply_pca.py \
        "${uasr_stats_dir}/${split}/collect_feats/split${nj}/JOB/feats.scp" \
        --split ${split} \
        --save_dir ${output_feats_dir}/precompute_pca$dim/JOB/ \
        --pca_path ${output_feats_dir}/pca/${dim}_pca \
        --batch_size 1048000

    echo "Merging clusters on ${split}"
    ${cmd} JOB=1:${nj} ${logdir}/merge_clusters_${split}.JOB.log \
        python pyscripts/feats/merge_clusters.py \
          ${output_feats_dir}/precompute_pca$dim/JOB \
          --cluster_dir ${output_feats_dir}/CLUS${num_clusters}/JOB \
          --split ${split} \
          --save_dir ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean/JOB \
          --pooling mean

    root="$(pwd)/"
    for n in $(seq ${nj}); do
        cut -d ' ' -f1 "${uasr_stats_dir}/${split}/collect_feats/split${nj}/${n}/feats.scp" | \
            paste -d ' ' - ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean/${n}/${split}.lengths_pure > \
            ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean/${n}/${split}.lengths
    done

    [ -f "${uasr_stats_dir}/${split}/speech_shape" ] && rm "${uasr_stats_dir}/${split}/speech_shape"
    for n in $(seq ${nj}); do
        cat ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean/${n}/${split}.lengths >> "${uasr_stats_dir}/${split}/speech_shape"
    done
    
    echo "Averaging ${split}"
    ${cmd} JOB=1:${nj} ${logdir}/merge_pca_${split}.JOB.log \
        python pyscripts/feats/mean_pool_scp.py \
            ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean/JOB/ \
            --save_dir ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean_pooled/JOB \
            --split ${split} \
            --root ${root}

    mkdir -p ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean_pooled/${split}
    [ -f ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean_pooled/${split}/feats.scp ] && \
        rm ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean_pooled/${split}/feats.scp

    for n in $(seq ${nj}); do
        cat ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean_pooled/${n}/${split}/feats.scp >> \
            ${output_feats_dir}/precompute_pca${dim}_cls${num_clusters}_mean_pooled/${split}/feats.scp
    done
done
