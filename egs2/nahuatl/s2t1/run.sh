#!/usr/bin/env bash
# Nahuatl ASR — OWSM v4 fine-tuning recipe
# Thin wrapper around egs2/TEMPLATE/s2t1/s2t.sh.
#
# The three regional dialects (Hidalgo, Orizaba-Zongolica, Zacatlan-Tepetzintla)
# are distinguished by *prompt-conditioning*: the dialect is supplied through the
# decoder prompt (text.prev = "nahuatl <region>") while the transcript uses
# OWSM's existing <na> language slot. This needs no vocabulary surgery — training
# starts from the released OWSM checkpoint and reuses its BPE model and token
# list unchanged.
#
# Prerequisites (one-time setup):
#   1. Download espnet/owsm_v4_medium_1B from HuggingFace into
#      "${MODEL_CACHE_DIR}/owsm_v4_medium_1B/" (MODEL_CACHE_DIR is set in path.sh;
#      it defaults to <repo_root>/../model_cache). The download must include the
#      exp/ checkpoint + config and data/token_list/bpe_unigram50000/bpe.model.
#   2. Build the HuggingFace dataset (see ../README.md) and run local/data.sh
#      (Stage 1) to produce the Kaldi data dirs.
#
# Quick usage:
#   bash run.sh                              # data prep + stats + train
#   bash run.sh --stage 11 --stop_stage 11   # train only (stage 10 already done)
#   bash run.sh --stage 12 --stop_stage 13   # decode + score the test sets
#   max_epoch=1 num_iters_per_epoch=20 log_interval=1 \
#       bash run.sh --stage 11 --stop_stage 11   # quick debug
set -euo pipefail

RECIPE_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
cd "$RECIPE_DIR"
. cmd.sh
. path.sh

# ── Recipe configuration ────────────────────────────────────────────────────
train_set="nahuatl_train"
valid_set="nahuatl_valid"
# The three single-region test sets. With prompt-conditioning every utterance is
# decoded with the same <na> language symbol, so a single merged test set would
# also decode correctly; we keep them split only to report a per-dialect CER
# breakdown (local/decode.py also emits the combined score).
test_sets="nahuatl_hidalgo_test nahuatl_orizaba_zongolica_test nahuatl_zacatlan_tepetzintla_test"

s2t_config="conf/train.yaml"

# ── OWSM upstream assets (released checkpoint, BPE model, token list, stats) ──
OWSM_DIR=$(realpath "${MODEL_CACHE_DIR}/owsm_v4_medium_1B")
OWSM_EXP="$OWSM_DIR/exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000"
OWSM_CONFIG="$OWSM_EXP/config.yaml"
OWSM_BPE="$OWSM_DIR/data/token_list/bpe_unigram50000/bpe.model"
OWSM_FEATS_STATS="$OWSM_DIR/exp/s2t_stats_raw_bpe50000/train/feats_stats.npz"

# s2t.sh (stages 5-6 skipped) expects the BPE model and token list under
# data/token_list/bpe_unigram50000/. Reuse OWSM's unchanged: symlink the BPE
# model and materialize tokens.txt from the OWSM config's embedded token_list.
TOKEN_LIST_DIR="data/token_list/bpe_unigram50000"
if [ ! -f "$OWSM_BPE" ]; then
    echo "ERROR: OWSM BPE model not found at $OWSM_BPE" >&2
    echo "Download espnet/owsm_v4_medium_1B into MODEL_CACHE_DIR (see header)." >&2
    exit 1
fi
mkdir -p "$TOKEN_LIST_DIR"
ln -sf "$(realpath "$OWSM_BPE")" "$TOKEN_LIST_DIR/bpe.model"
if [ ! -s "$TOKEN_LIST_DIR/tokens.txt" ]; then
    if [ ! -f "$OWSM_CONFIG" ]; then
        echo "ERROR: OWSM config not found at $OWSM_CONFIG" >&2
        echo "It carries the token_list used to build tokens.txt." >&2
        exit 1
    fi
    echo "Extracting token list from $OWSM_CONFIG"
    python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$OWSM_CONFIG'))
toks = cfg['token_list']
if isinstance(toks, str):
    toks = [l.rstrip('\n') for l in open(toks)]
with open('$TOKEN_LIST_DIR/tokens.txt', 'w') as f:
    f.write('\n'.join(toks) + '\n')
print(f'Wrote {len(toks)} tokens to $TOKEN_LIST_DIR/tokens.txt')
"
fi

# Fine-tuning must reuse OWSM's global-MVN feature statistics (the model was
# pretrained with them). collect_stats (stage 10) with global_mvn runs
# aggregate_stats_dirs, which WRITES train/feats_stats.npz into the stats dir,
# and stage 11 hard-wires --normalize_conf stats_file=<stats_dir>/train/feats_stats.npz
# on the command line (a config override cannot redirect it). So the only way to
# train on OWSM's statistics is to overwrite that file AFTER stage 10 produces
# the per-utterance shape files and BEFORE stage 11 reads it — hence the split
# around collect_stats in the delegation below.
STATS_TRAIN_DIR="exp/s2t_stats_raw_bpe50000/train"
install_feats_stats() {
    # Called only on training paths. Require the pretrained stats: without them
    # stage 11 would silently fine-tune on the wrong (Nahuatl-only) global-MVN
    # statistics stage 10 produced.
    if [ ! -f "$OWSM_FEATS_STATS" ]; then
        echo "ERROR: pretrained feature statistics not found at" >&2
        echo "  $OWSM_FEATS_STATS" >&2
        echo "Fine-tuning requires OWSM's global-MVN stats; download the" \
             "pretrained model into MODEL_CACHE_DIR (see header) before training." >&2
        exit 1
    fi
    mkdir -p "$STATS_TRAIN_DIR"
    ln -sf "$(realpath "$OWSM_FEATS_STATS")" "$STATS_TRAIN_DIR/feats_stats.npz"
}

# Debug overrides: set these env vars to limit training
# e.g.: max_epoch=1 num_iters_per_epoch=20 log_interval=1 bash run.sh --stage 11 --stop_stage 11
_s2t_args=""
[ -n "${max_epoch:-}" ]            && _s2t_args+=" --max_epoch $max_epoch"
[ -n "${num_iters_per_epoch:-}" ]  && _s2t_args+=" --num_iters_per_epoch $num_iters_per_epoch"
[ -n "${log_interval:-}" ]         && _s2t_args+=" --log_interval $log_interval"

# ── Delegate to TEMPLATE s2t.sh ─────────────────────────────────────────────
# init_param (the released OWSM checkpoint) is set in conf/train.yaml so it stays
# out of the experiment tag, keeping the exp dir name clean.
s2t_opts=(
    --use_lm false
    --ngpu 1
    --feats_type raw
    --audio_format wav
    --fs 16k
    --min_wav_duration 0.1
    --max_wav_duration 30.5
    --token_type bpe
    --nbpe 50000
    --s2t_config "${s2t_config}"
    --train_set "${train_set}"
    --valid_set "${valid_set}"
    --test_sets "${test_sets}"
    --skip_stages "5 6"
)
[ -n "${_s2t_args}" ] && s2t_opts+=(--s2t_args "${_s2t_args}")

# Requested stage range (run.sh defaults: data prep through training).
start_stage=1
stop_stage=11
_prev=""
for _a in "$@"; do
    # Accept both "--stage VALUE" and "--stage=VALUE" forms (s2t.sh takes both).
    case "$_a" in
        --stage=*)      start_stage="${_a#--stage=}" ;;
        --stop_stage=*) stop_stage="${_a#--stop_stage=}" ;;
    esac
    [ "$_prev" = "--stage" ]      && start_stage="$_a"
    [ "$_prev" = "--stop_stage" ] && stop_stage="$_a"
    _prev="$_a"
done

# Decode + score (stage 12+) needs the dialect prompt (text.prev) fed to
# inference, which s2t.sh's decode path does not support. local/decode.py handles
# it: one <na> language symbol for all utterances, each conditioned on its own
# text.prev, scored per set and combined. Require exactly one matching exp dir so
# an older/unrelated run is never scored by filename order.
if [ "${start_stage}" -ge 12 ]; then
    mapfile -t _exps < <(ls -d exp/s2t_train_owsm_v4_nahuatl_raw_bpe50000*/ 2>/dev/null)
    if [ "${#_exps[@]}" -ne 1 ]; then
        echo "ERROR: expected exactly one exp dir matching" \
             "exp/s2t_train_owsm_v4_nahuatl_raw_bpe50000*, found ${#_exps[@]}:" \
             "${_exps[*]:-<none>}" >&2
        exit 1
    fi
    exec python3 local/decode.py \
        --exp_dir "${_exps[0]%/}" \
        --decode_config conf/decode.yaml \
        --test_sets "${test_sets}"
fi

if [ "${start_stage}" -le 10 ] && [ "${stop_stage}" -ge 11 ]; then
    # Training run that also builds stats: split around collect_stats (stage 10)
    # so its aggregate step cannot clobber the pretrained feats_stats.npz.
    # Trailing --stage/--stop_stage win over any in "$@".
    ./s2t.sh "${s2t_opts[@]}" "$@" --stop_stage 10          # data + collect_stats
    install_feats_stats                                     # pretrained global-MVN
    ./s2t.sh "${s2t_opts[@]}" "$@" --stage 11 --stop_stage 11
elif [ "${start_stage}" -le 11 ] && [ "${stop_stage}" -ge 11 ]; then
    # Training only (stage 10 already done in a prior run): install the pretrained
    # stats (overwriting any Nahuatl stats from that stage 10), then delegate.
    install_feats_stats
    ./s2t.sh "${s2t_opts[@]}" "$@"
else
    # Data-prep-only or other non-training range: stats are untouched.
    ./s2t.sh "${s2t_opts[@]}" "$@"
fi
