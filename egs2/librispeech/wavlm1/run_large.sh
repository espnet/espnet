#!/usr/bin/env bash
# WavLM LARGE (torchaudio wavlm_large: 1024/24/16/4096, pre-LN, gated relative
# position bias) -- 315.45 M backbone parameters, verified identical to
# torchaudio.models.wavlm_large().
#
# Batch sizing and torch.compile come from a measured 8 x H100 sweep; see the
# config headers and local/bench/ for the raw numbers.
set -e
set -u
set -o pipefail

. ./db.sh

train_start_iter=0
train_stop_iter=2

n_clusters_iter0=100
n_clusters_iter1=500
n_clusters_iter2=500

feature_iter0="mfcc";         layer_iter0="0"
feature_iter1="espnet_wavlm"; layer_iter1="6"
feature_iter2="espnet_wavlm"; layer_iter2="9"

# Multilingual pre-training corpus: 34 corpus/language dirs built by
# local/data_pretraining.sh from /mnt/weka/data/tagger_data/pretraining.
train_set="pretrain_train"
valid_set="pretrain_dev"

# NOTE: iterations 0 and 1 share the it0 config (100- vs 500-cluster targets are
# handled by --n_clusters); only label_downsampling differs, so iteration 1 uses
# the it2 config, which is the 500-cluster / model-frame-rate variant.
train_config_iter0=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it0.yaml
train_config_iter1=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it2.yaml
train_config_iter2=conf/tuning/train_ssl_torchaudiowavlm_large_960h_pretrain_it2.yaml

# ON as of the epoch-3 restart -- this is an EXPERIMENT, resumable either way.
# Trainer._maybe_compile now compiles the WHOLE model (it used to compile each
# ModuleList entry in place, which wrote `_orig_mod.` into checkpoint keys and
# made them unloadable; whole-model compile leaves `model` untouched, so the
# checkpoints stay clean). Verified on 2x H100 with the real 316.3 M-param model
# under DDP(find_unused_parameters=True): 30 steps, 5 batch shapes, no deadlock.
#
# WHAT WE DO NOT KNOW: why this job deadlocked three times at epoch 2 batch
# 13,100. The zero-masked-frame explanation was WRONG (see the note in the it0
# config), so torch.compile is NOT established as an innocent bystander in those
# hangs -- two of the three happened with it on. Baseline to beat is 0.60 s/step
# eager; the -25.3% figure is a LibriSpeech measurement that has never
# reproduced here. If it deadlocks or is slower at real batch_bins, set this
# back to false and resume -- nothing is lost, the epoch-3 checkpoint is clean.
use_torch_compile=true

# --num_splits_ssl 8 AND --lazy_km_labels together. Each fixes a different
# term of the same OOM, and neither is sufficient alone -- measured per rank on
# the full 62.5 M-utterance corpus:
#     text.km          147.9 GB  -> 0.03 GB   (lazy_km_labels)
#     wav.scp           16.3 GB \
#     speech_shape      19.6 GB  |-> 50.0 GB, only num_splits_ssl shrinks these
#     text_shape.word   14.0 GB /
# With lazy labels but num_splits_ssl 1, those 50 GB x 8 ranks plus model,
# optimiser and 64 dataloader workers still hit OUT_OF_MEMORY at 843 GB.
# At 8 splits the file memory is 6.3 GB/rank (50 GB across ranks).
#
# MEASURED, correcting an earlier guess: the per-split reload costs ~280 s
# whatever N is. It does NOT scale with shard size -- N=16 (0.8 GB shards) cost
# 247 s and N=8 (1.6 GB shards) cost 283 s, i.e. doubling the data added 15%.
# The cost is fixed overhead (rebuilding the batch sampler, re-spawning 64
# dataloader workers across 8 ranks), not bytes read, so making the labels lazy
# did not make reloads cheap. N therefore trades against MEMORY only, and the
# reload cost is amortised solely by num_iters_per_epoch (see the config
# headers): at 40000 iters, 8 reloads x 280 s is 8% of a 4.6 h epoch. That also removes
# MultipleIterFactory's per-split dataset rebuild, measured at ~247 s x 16 per
# epoch (22% of wall clock). Verified on the real 62.5 M-utterance file: open
# 4 ms, 0.03 GB resident, 62 us per lookup, values bit-identical through
# ESPnetDataset + CommonPreprocessor.
#
# Kept for reference -- num_splits_ssl WAS required before that: ESPnetDataset loads a "text"
# input eagerly via read_2columns_text (dataset.py:364) -- the whole file into a
# dict, per rank. text.km for this corpus is 147.9 GB, so 8 ranks tried to hold
# well over 1 TB and the job was OOM-killed at 900 GB after 18 min
# (State=OUT_OF_MEMORY, MaxRSS 943,698,232K on a 942 GB node).
# --num_splits_ssl 16 splits wav.scp, text.km and both shape files into 16
# chunks and sets --multiple_iterator true, so each rank holds ~9.2 GB of
# labels instead of 147.9 GB. The split is one-time and cached behind a .done
# marker. Note each epoch then draws from ONE split, i.e. 1/16 of the corpus,
# so valid loss is noisier epoch to epoch (the dev set itself is unsplit, so
# comparisons stay valid).
#
# storage_save_mode MUST be true at this data scale. perform_kmeans.sh defaults
# it to false, which makes stage 1 dump features for the whole training set
# rather than the --portion_km subset: 8.4 TB of MFCC for iteration 0 and
# 110 TB of 1024-dim WavLM features for iterations 1-2. With it true, stage 1
# dumps only the 299 h subset (17 GB / 220 GB) and stage 3 passes
# --online_feature_extract, computing features on the fly.
#
# percent caps what stage 2 fits k-means on. learn_kmeans.py holds every
# selected frame in RAM at once, and perform_kmeans.sh used to hardcode
# --percent -1 (all of them): 1024-dim WavLM features over the 299 h subset is
# 220 GB, which risks OOM once sklearn's check_array copies. 0.25 brings that to
# ~55 GB and still fits 500 clusters on 13.4 M frames (~27 k per cluster).
# It samples utterances uniformly, so per-language balance is preserved.
#
# split_by_duration: split_scp.pl gives every label-dumping job an equal
# NUMBER of utterances. Because utt ids sort as <corpus>-<lang>-..., each split
# lands inside one corpus, and mean utterance length ranges from 3.2 s
# (mdcc) to 30 s (voxpopuli). Measured on the real corpus at nj=128: the
# heaviest split held 3897 h against the lightest 374 h, a 10.4x spread, and the
# stage runs at the pace of the worst straggler -- 37/128 jobs done was only 16%
# of the audio, tracking to ~8 h. Cutting on cumulative samples instead gives
# every job ~1169 h and should bring that to ~1.2 h.
#
# The km_* knobs are stage-2 MiniBatchKMeans settings, and they matter far more
# than they look. learn_kmeans.py defaults to max_iter 100 with tol 0, i.e. 100
# full passes over every frame and no convergence test:
#   total steps = max_iter * n_frames / batch_size
#   iter 0   (26.9 M frames,  39-dim, 100 clust): 267,409 steps -> 16 h  (measured)
#   iter 1/2 (13.4 M frames, 1024-dim, 500 clust):                ~45 days (projected;
#            per-step cost scales with batch_size * dim * nclusters)
# Measured convergence: after 0.8 of ONE pass the ewa inertia was already within
# 1.7% of its extrapolated asymptote (2048 -> 1975, asymptote ~1942), improving
# geometrically at x0.879 per 200 steps. 10 passes is ample; 100 is waste.
# batch_size 100000 cuts per-step Python/numpy overhead (per-pass work is
# unchanged), tol 1e-4 re-enables the convergence test that tol=0 disables, and
# n_init 3 replaces 20 redundant initialisations.
#
# batch_bins is in raw samples and is per-job (NumElementsBatchSampler over
# utt2num_samples). It defaults to 1 in dump_km_label.py -- i.e. one utterance
# per forward pass -- which would be ruinous over 62 M utterances. 8 M samples
# is ~500 s of audio per batch; the readers run under torch.inference_mode(),
# so nothing is retained and this is comfortable on one 80 GB GPU.

# Shared tree: data lives here (see local/data_pretraining.sh) and checkpoints
# should too, so teammates in the `research` group can read and resume them.
# NOTE: --dumpdir/--expdir do not cover the k-means token list, which
# wavlm.sh:352 builds at the recipe-relative data/${lang}_token_list_*.
./wavlm.sh \
    --ngpu 8 \
    --dumpdir /mnt/weka/data/wavlm_expts/dump \
    --expdir  /mnt/weka/data/wavlm_expts/exp \
    --num_nodes 1 \
    --lang "multi" \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --num_splits_ssl 8 \
    --lazy_km_labels true \
    --max_wav_duration 30.01 \
    --train_configs "${train_config_iter0} ${train_config_iter1} ${train_config_iter2}" \
    --n_clusters "${n_clusters_iter0} ${n_clusters_iter1} ${n_clusters_iter2}" \
    --features_km "${feature_iter0} ${feature_iter1} ${feature_iter2}" \
    --layers_km "${layer_iter0} ${layer_iter1} ${layer_iter2}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --portion_km 0.002 \
    --kmeans_opts "--storage_save_mode true --batch_bins 8000000 --percent 0.25 --split_by_duration true \
                   --km_max_iter 10 --km_batch_size 100000 --km_tol 1e-4 \
                   --km_max_no_improvement 20 --km_n_init 3" \
    --gpu_dump_feature true \
    --wavlm_args "--use_torch_compile ${use_torch_compile}" "$@"
