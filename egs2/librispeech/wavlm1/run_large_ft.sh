#!/usr/bin/env bash
# WavLM LARGE, continued pretraining from torchaudio's pretrained weights.
#
# DIFFERENT SETUP from run_large.sh. That script runs the 3-iteration HuBERT-style
# pipeline from scratch: MFCC k-means -> train -> recluster on own features ->
# train -> recluster -> train. This script skips all of that. It:
#
#   1. clusters layer-9 features of the PRETRAINED WavLM-large (500 clusters,
#      the layer/cluster count HuBERT and espnet use for iteration 2), and
#   2. trains WavLM-large INITIALISED from those same pretrained weights, once.
#
# Rationale: the pretrained model has already seen 94k h, so its layer-9 features
# are a far better clustering target than MFCCs, and its weights are a far better
# starting point than random. Iterations 0 and 1 exist only to bootstrap past bad
# targets, which we do not need.
#
# HOW THE TEACHER IS FOUND: wavlm.sh derives iteration 2's teacher as
#   ${expdir}/wavlm_iter1_$(basename <2nd train_config> .yaml)_raw/valid.loss.best.pth
# so passing the it2 config as the 2nd element points it at the checkpoint holding
# the torchaudio weights in espnet format. No code change needed.
set -e
set -u
set -o pipefail

. ./db.sh

# One "iteration", numbered 2 so the recipe uses the layer-9 / 500-cluster path.
train_start_iter=2
train_stop_iter=2

train_set="pretrain_train"
valid_set="pretrain_dev"

IT0=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0.yaml
IT2=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it2.yaml
FT=conf/tuning/train_ssl_torchaudiowavlm_large_ft.yaml

# num_threads 8, NOT 4. On this cluster MaxMemPerCPU = DefMemPerCPU = 14680M,
# so memory can ONLY be raised by asking for more CPUs -- num_threads controls
# both the thread count and the memory ceiling. At 4 threads the fit gets 57 GB
# and is OOM-killed: load_feature_shard holds all 51 GB of dumped features plus
# a 13 GB sampled copy, peaking ~64 GB. 8 threads gives 115 GB. (learn_kmeans
# reports threads=1 regardless, so the extra CPUs are bought for their memory.)
#
# portion_km 0.0005, not the 0.002 used elsewhere: learn_kmeans.py's
# load_feature_shard reads EVERY dumped feature into a list before --percent
# samples it, so the peak is set by the dump size, not by percent. At 1024 dims
# 0.002 (299 h) dumps 220 GB and peaks ~275 GB; 0.0005 (75 h) dumps ~55 GB and
# peaks ~69 GB. 75 h is still 13.5 M frames, ~27 k per cluster for 500 clusters.
./wavlm.sh \
    --ngpu 8 \
    --num_nodes 1 \
    --dumpdir /mnt/weka/data/wavlm_expts/dump \
    --expdir  /mnt/weka/data/wavlm_expts/exp \
    --lang "multi" \
    --train_start_iter "${train_start_iter}" \
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --num_splits_ssl 8 \
    --lazy_km_labels true \
    --max_wav_duration 30.01 \
    --train_configs "${IT0} ${IT2} ${FT}" \
    --n_clusters "100 500 500" \
    --features_km "mfcc espnet_wavlm espnet_wavlm" \
    --layers_km "0 6 9" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --portion_km 0.0005 \
    --gpu_dump_feature true \
    --kmeans_opts "--storage_save_mode true --batch_bins 8000000 --percent 0.25 --split_by_duration true \
                   --km_max_iter 10 --km_batch_size 100000 --km_tol 1e-4 \
                   --km_max_no_improvement 20 --km_n_init 3 --num_threads 8 \
                   --assume_sorted_splits true --fast_label_finalize true \
                   --token_count_sample_utts 2000" \
    "$@"
