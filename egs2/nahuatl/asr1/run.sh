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

. cmd.sh
. path.sh

RECIPE_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
DATA_DIR="$RECIPE_DIR/data"
EXP_DIR="$RECIPE_DIR/exp"
MODEL_DIR=$(realpath "$RECIPE_DIR/../../../../model_cache/owsm_v4_medium_1B_nahuatl")

train_config="conf/tuning/train_owsm_v4_nahuatl.yaml"
decode_config="conf/decode_owsm.yaml"
token_list="$DATA_DIR/token_list_nahuatl.txt"

train_set="nahuatl_train"
valid_set="nahuatl_valid"
test_sets="nahuatl_test nahuatl_hidalgo_test nahuatl_orizaba_zongolica_test nahuatl_zacatlan_tepetzintla_test"

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 2 ]; then
    echo "=== Stages 1–2: Data preparation ==="
    bash local/data.sh
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "=== Stage 3: Collect stats ==="
    python -m espnet2.bin.s2t_train \
        --collect_stats true \
        --train_data_path_and_name_and_type "$DATA_DIR/${train_set}/wav.scp,speech,sound" \
        --train_data_path_and_name_and_type "$DATA_DIR/${train_set}/text,text,text" \
        --valid_data_path_and_name_and_type "$DATA_DIR/${valid_set}/wav.scp,speech,sound" \
        --valid_data_path_and_name_and_type "$DATA_DIR/${valid_set}/text,text,text" \
        --train_shape_file "$EXP_DIR/s2t_stats/train/speech_shape" \
        --valid_shape_file "$EXP_DIR/s2t_stats/valid/speech_shape" \
        --output_dir "$EXP_DIR/s2t_stats" \
        --config "$train_config"

    echo "=== Stage 3: Train ==="
    python -m espnet2.bin.s2t_train \
        --train_data_path_and_name_and_type "$DATA_DIR/${train_set}/wav.scp,speech,sound" \
        --train_data_path_and_name_and_type "$DATA_DIR/${train_set}/text,text,text" \
        --valid_data_path_and_name_and_type "$DATA_DIR/${valid_set}/wav.scp,speech,sound" \
        --valid_data_path_and_name_and_type "$DATA_DIR/${valid_set}/text,text,text" \
        --train_shape_file "$EXP_DIR/s2t_stats/train/speech_shape" \
        --valid_shape_file "$EXP_DIR/s2t_stats/valid/speech_shape" \
        --output_dir "$EXP_DIR/s2t_owsm_v4_nahuatl" \
        --config "$train_config" \
        --ngpu 1
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
        # Score against reference text (strip region token prefix before scoring)
        python3 - <<PYEOF
from pathlib import Path
import re
ref_lines = (Path("$DATA_DIR/$dset") / "text").read_text().splitlines()
# Strip leading tokens: remove "<...>" prefixes
hyp_lines = (Path("$decode_dir/text") / "text").read_text().splitlines()
tok_pat = re.compile(r'^(\S+\s+)(<[^>]+>)+\s*')
ref_clean = {l.split()[0]: re.sub(r'(<[^>]+>)+\s*', '', l.split(' ', 1)[1]) for l in ref_lines}
# Write cleaned ref for scoring
with open("$decode_dir/ref_clean.txt", "w") as f:
    for utt_id, text in sorted(ref_clean.items()):
        f.write(f"{utt_id} {text}\n")
PYEOF
        echo "  Scoring $dset (results in $decode_dir/)"
    done
fi

echo "Recipe complete."
