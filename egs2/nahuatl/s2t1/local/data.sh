#!/usr/bin/env bash
# Prepare all 9 HF splits into Kaldi data dirs, then merge by set.
#
# The dialect is carried by the prompt (text.prev = "nahuatl <region>"), so the
# per-region data dirs already differ only by that prompt; merging is a plain
# concatenation of the six per-utterance files.
set -euo pipefail

RECIPE_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
# shellcheck disable=SC1091
. "$RECIPE_DIR/path.sh"
# shellcheck disable=SC1091
. "$RECIPE_DIR/db.sh"

# The Nahuatl HF dataset location is registered as NAHUATL in db.sh.
HF_DATA_DIR="${NAHUATL:?Set NAHUATL in db.sh to the Nahuatl HF dataset path}"

DATA_DIR="$RECIPE_DIR/data"
WAV_BASE="$DATA_DIR/wav"

run_prep() {
    local hf_split=$1 kaldi_name=$2 prompt=$3
    local out="$DATA_DIR/$kaldi_name"
    local wav="$WAV_BASE/$kaldi_name"
    # A split is complete only when all Kaldi files are present and non-empty; a
    # partial prep (e.g. crash after wav.scp) must not be skipped.
    if [[ -s "$out/wav.scp" && -s "$out/text" && -s "$out/text.prev" \
          && -s "$out/text.ctc" && -s "$out/utt2spk" && -s "$out/spk2utt" ]]; then
        echo "  skip $kaldi_name (exists)"; return
    fi
    echo "  $hf_split -> $kaldi_name"
    # Prepare into a temp dir and rename into place only after success, so an
    # interrupted run never leaves a half-written split behind.
    rm -rf "$out.tmp"
    python "$RECIPE_DIR/local/data_prep.py" \
        --hf_data_dir "$HF_DATA_DIR" \
        --split       "$hf_split" \
        --output_dir  "$out.tmp" \
        --wav_dir     "$wav" \
        --lang_prompt "$prompt"
    for f in wav.scp text text.prev text.ctc utt2spk spk2utt; do
        [[ -s "$out.tmp/$f" ]] || { echo "ERROR: $kaldi_name prep produced no $f" >&2; exit 1; }
    done
    rm -rf "$out"; mv "$out.tmp" "$out"
}

echo "=== Stage 1: Per-region data prep ==="
run_prep hidalgo-train              nahuatl_hidalgo_train              "nahuatl hidalgo"
run_prep hidalgo-val                nahuatl_hidalgo_valid              "nahuatl hidalgo"
run_prep hidalgo-test               nahuatl_hidalgo_test               "nahuatl hidalgo"
run_prep orizaba-zongolica-train    nahuatl_orizaba_zongolica_train    "nahuatl orizaba zongolica"
run_prep orizaba-zongolica-val      nahuatl_orizaba_zongolica_valid    "nahuatl orizaba zongolica"
run_prep orizaba-zongolica-test     nahuatl_orizaba_zongolica_test     "nahuatl orizaba zongolica"
run_prep zacatlan-tepetzintla-train nahuatl_zacatlan_tepetzintla_train "nahuatl zacatlan tepetzintla"
run_prep zacatlan-tepetzintla-val   nahuatl_zacatlan_tepetzintla_valid "nahuatl zacatlan tepetzintla"
run_prep zacatlan-tepetzintla-test  nahuatl_zacatlan_tepetzintla_test  "nahuatl zacatlan tepetzintla"

echo "=== Stage 2: Merge by set ==="
# dotted filenames are the ESPnet s2t.sh convention for utt_extra_files; the
# data names become text_prev / text_ctc (s2t.sh maps '.' -> '_').
for setname in train valid test; do
    out="$DATA_DIR/nahuatl_${setname}"
    mkdir -p "$out"
    for f in wav.scp text text.prev text.ctc utt2spk; do
        rm -f "$out/$f"
        for region in hidalgo orizaba_zongolica zacatlan_tepetzintla; do
            cat "$DATA_DIR/nahuatl_${region}_${setname}/$f" >> "$out/$f"
        done
        sort -k1 -o "$out/$f" "$out/$f"
    done

    # Rebuild spk2utt from merged utt2spk
    python3 -c "
import collections
from pathlib import Path
p = Path('$out')
spk2utt = collections.defaultdict(list)
for line in (p / 'utt2spk').read_text().splitlines():
    utt, spk = line.split()
    spk2utt[spk].append(utt)
with open(p / 'spk2utt', 'w') as f:
    for spk, utts in sorted(spk2utt.items()):
        f.write(spk + ' ' + ' '.join(sorted(utts)) + '\n')
"
    echo "  nahuatl_${setname}: $(wc -l < "$out/wav.scp") utterances"
done
echo "Done."
