#!/usr/bin/env bash
# Nahuatl ASR — OWSM v4 fine-tuning. Thin wrapper around TEMPLATE/s2t1/s2t.sh.
# The three dialects are distinguished by prompt-conditioning (text.prev =
# "nahuatl <region>") on OWSM's existing <na> slot, so training starts from the
# released checkpoint with no vocabulary surgery. See README.md for setup.
#
# Usage:
#   ./local/data.sh                          # data prep + token list (run first)
#   ./run.sh                                 # collect_stats + fine-tune
#   ./run.sh --stage 12 --stop_stage 13      # decode + score the test sets
#   max_epoch=1 num_iters_per_epoch=20 log_interval=1 \
#       ./run.sh --stage 11 --stop_stage 11  # quick debug
set -euo pipefail

RECIPE_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
cd "$RECIPE_DIR"
. cmd.sh
. path.sh

train_set="nahuatl_train"
valid_set="nahuatl_valid"
# Per-dialect test sets, only to report a per-dialect CER breakdown; with
# prompt-conditioning all three decode with the same <na> symbol in one pass.
test_sets="nahuatl_hidalgo_test nahuatl_orizaba_zongolica_test nahuatl_zacatlan_tepetzintla_test"

# Resolve the requested stage range (accept "--stage N" and "--stage=N").
start_stage=1; stop_stage=11; _prev=""
for _a in "$@"; do
    case "$_a" in
        --stage=*)      start_stage="${_a#--stage=}" ;;
        --stop_stage=*) stop_stage="${_a#--stop_stage=}" ;;
    esac
    [ "$_prev" = "--stage" ]      && start_stage="$_a"
    [ "$_prev" = "--stop_stage" ] && stop_stage="$_a"
    _prev="$_a"
done

# Decode + score (stage 12+) feeds each utterance its own text.prev prompt, which
# s2t.sh's decode path does not support; local/decode.py handles it end to end.
if [ "${start_stage}" -ge 12 ]; then
    exec python3 local/decode.py --test_sets "${test_sets}"
fi

# Fine-tuning must reuse OWSM's global-MVN statistics. collect_stats (stage 10)
# writes its own feats_stats.npz into the stats dir and stage 11 reads it from a
# path fixed on the s2t.sh command line, so the pretrained stats are swapped in
# after stage 10 completes and before stage 11 — hence the split below.
OWSM_FEATS_STATS="$(realpath "${MODEL_CACHE_DIR}")/owsm_v4_medium_1B/exp/s2t_stats_raw_bpe50000/train/feats_stats.npz"
STATS_TRAIN_DIR="exp/s2t_stats_raw_bpe50000/train"
install_feats_stats() {
    [ -f "$OWSM_FEATS_STATS" ] || { echo "ERROR: OWSM feats_stats not found at $OWSM_FEATS_STATS (see README.md)" >&2; exit 1; }
    mkdir -p "$STATS_TRAIN_DIR"
    ln -sf "$OWSM_FEATS_STATS" "$STATS_TRAIN_DIR/feats_stats.npz"
}

# Debug overrides (see usage above).
_s2t_args=""
[ -n "${max_epoch:-}" ]           && _s2t_args+=" --max_epoch $max_epoch"
[ -n "${num_iters_per_epoch:-}" ] && _s2t_args+=" --num_iters_per_epoch $num_iters_per_epoch"
[ -n "${log_interval:-}" ]        && _s2t_args+=" --log_interval $log_interval"

# init_param (the released OWSM checkpoint) lives in conf/train.yaml, keeping it
# out of the experiment tag.
s2t_opts=(
    --use_lm false --ngpu 1
    --feats_type raw --audio_format wav --fs 16k
    --min_wav_duration 0.1 --max_wav_duration 30.5
    --token_type bpe --nbpe 50000
    --s2t_config conf/train.yaml
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}"
    --skip_stages "5 6"
)
[ -n "${_s2t_args}" ] && s2t_opts+=(--s2t_args "${_s2t_args}")

if [ "${start_stage}" -le 10 ] && [ "${stop_stage}" -ge 11 ]; then
    ./s2t.sh "${s2t_opts[@]}" "$@" --stop_stage 10   # data + collect_stats
    install_feats_stats                              # swap in OWSM's global-MVN
    ./s2t.sh "${s2t_opts[@]}" "$@" --stage 11 --stop_stage 11
elif [ "${start_stage}" -le 11 ] && [ "${stop_stage}" -ge 11 ]; then
    install_feats_stats
    ./s2t.sh "${s2t_opts[@]}" "$@"
else
    ./s2t.sh "${s2t_opts[@]}" "$@"
fi
