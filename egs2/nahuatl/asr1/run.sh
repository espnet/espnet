#!/usr/bin/env bash
# ESPnet2 Nahuatl OWSM v4 fine-tuning recipe
# Stages:
#   1  Data prep (per-region Kaldi dirs)
#   2  Merge splits into nahuatl_{train,valid,test}
#   3  Collect stats + train (requires GPU)
#   4  Decode (requires GPU)
#   5  Score (CER/WER)
set -euo pipefail

stage=${stage:-1}
stop_stage=${stop_stage:-5}

# Debug overrides: set these env vars to limit training for quick smoke tests
# e.g.: max_epoch=1 num_iters_per_epoch=50 stage=3 stop_stage=3 bash run.sh
max_epoch=${max_epoch:-}
num_iters_per_epoch=${num_iters_per_epoch:-}

. cmd.sh
. path.sh

RECIPE_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
cd "$RECIPE_DIR"
DATA_DIR="$RECIPE_DIR/data"
EXP_DIR="$RECIPE_DIR/exp"
MODEL_DIR=$(realpath "$RECIPE_DIR/../../../../model_cache/owsm_v4_medium_1B_nahuatl")

train_config="conf/tuning/train_owsm_v4_nahuatl.yaml"
decode_config="conf/decode_owsm.yaml"
token_list="$DATA_DIR/token_list_nahuatl.txt"

train_set="nahuatl_train"
valid_set="nahuatl_valid"
test_sets="nahuatl_test nahuatl_hidalgo_test nahuatl_orizaba_zongolica_test nahuatl_zacatlan_tepetzintla_test"

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "=== Stages 1–2: Data preparation ==="
    bash local/data.sh
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    # Stage 3 prerequisite: patched checkpoint must exist
    if [ ! -f "$MODEL_DIR/valid.loss.best.pth" ]; then
        echo "ERROR: Patched checkpoint not found at $MODEL_DIR/valid.loss.best.pth"
        echo "Run: python local/init_new_tokens.py \\"
        echo "  --src_ckpt <path/to/owsm_v4_medium_1B/valid.loss.best.pth> \\"
        echo "  --out_ckpt $MODEL_DIR/valid.loss.best.pth \\"
        echo "  --n_new 3"
        exit 1
    fi

    # collect_stats: skip if shape files already exist
    if [ ! -f "$EXP_DIR/s2t_stats/train/speech_shape" ]; then
        echo "=== Stage 3: Collect stats ==="
        python -m espnet2.bin.s2t_train \
            --collect_stats true \
            --ngpu 0 \
            --multiprocessing_distributed false \
            --dist_launcher None \
            --dist_world_size 1 \
            --output_dir "$EXP_DIR/s2t_stats" \
            --config "$train_config" \
            --token_list "$token_list"
    else
        echo "=== Stage 3: Collect stats — skipping (shape files exist) ==="
    fi

    echo "=== Stage 3: Train ==="
    python -m espnet2.bin.s2t_train \
        --output_dir "$EXP_DIR/s2t_owsm_v4_nahuatl" \
        --config "$train_config" \
        --init_param "$MODEL_DIR/valid.loss.best.pth" \
        --token_list "$token_list" \
        --ngpu 1 \
        ${max_epoch:+--max_epoch "$max_epoch"} \
        ${num_iters_per_epoch:+--num_iters_per_epoch "$num_iters_per_epoch"} \
        ${log_interval:+--log_interval "$log_interval"}
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    echo "=== Stage 4: Decode ==="
    for dset in $test_sets; do
        python -m espnet2.bin.s2t_inference \
            --output_dir "$EXP_DIR/s2t_owsm_v4_nahuatl/decode_${dset}" \
            --data_path_and_name_and_type "$DATA_DIR/${dset}/wav.scp,speech,sound" \
            --key_file "$DATA_DIR/${dset}/wav.scp" \
            --s2t_train_config "$EXP_DIR/s2t_owsm_v4_nahuatl/config.yaml" \
            --s2t_model_file "$EXP_DIR/s2t_owsm_v4_nahuatl/valid.loss.best.pth" \
            --config "$decode_config" \
            --ngpu 1
    done
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
    echo "=== Stage 5: Score ==="
    for dset in $test_sets; do
        decode_dir="$EXP_DIR/s2t_owsm_v4_nahuatl/decode_${dset}"
        python3 - "$DATA_DIR/$dset/text" "$decode_dir/1best_recog/text" \
                   "$decode_dir/ref_clean.txt" "$decode_dir/hyp_clean.txt" <<'PYEOF'
import sys, re

ref_path, hyp_path, ref_out, hyp_out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

tok_pat = re.compile(r'(<nah_hid>|<nah_ozg>|<nah_ztp>|<asr>|<notimestamps>)\s*')

def strip_tokens(line):
    parts = line.split(None, 1)
    if len(parts) < 2:
        return parts[0], ""
    utt_id, text = parts
    text = tok_pat.sub('', text).strip()
    return utt_id, text

ref_lines = open(ref_path).read().splitlines()
hyp_lines = open(hyp_path).read().splitlines()

ref_clean = {}
for line in ref_lines:
    uid, text = strip_tokens(line)
    ref_clean[uid] = text

hyp_clean = {}
for line in hyp_lines:
    uid, text = strip_tokens(line)
    hyp_clean[uid] = text

with open(ref_out, "w") as f:
    for uid in sorted(ref_clean):
        f.write(f"{uid} {ref_clean[uid]}\n")

with open(hyp_out, "w") as f:
    for uid in sorted(hyp_clean):
        f.write(f"{uid} {hyp_clean[uid]}\n")

# Compute CER using jiwer if available
common_ids = sorted(set(ref_clean) & set(hyp_clean))
refs = [ref_clean[uid] for uid in common_ids]
hyps = [hyp_clean[uid] for uid in common_ids]

try:
    import jiwer
    transform = jiwer.transforms.Compose([
        jiwer.transforms.RemoveMultipleSpaces(),
        jiwer.transforms.Strip(),
        jiwer.transforms.ReduceToSingleSentence(),
        jiwer.transforms.ReduceToListOfListOfChars(),
    ])
    cer = jiwer.wer(refs, hyps,
                    reference_transform=transform,
                    hypothesis_transform=transform)
    print(f"CER: {cer * 100:.2f}%  ({len(common_ids)} utterances)")
except ImportError:
    print("jiwer not available — skipping CER computation.")
    print("To score manually: pip install jiwer, then re-run Stage 5.")
    print(f"Cleaned ref: {ref_out}")
    print(f"Cleaned hyp: {hyp_out}")
PYEOF
        echo "  Scoring $dset done (results in $decode_dir/)"
    done
fi

echo "Recipe complete."
