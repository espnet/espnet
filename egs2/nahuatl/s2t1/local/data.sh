#!/usr/bin/env bash
# Prepare all 9 HF splits into Kaldi data dirs, then merge by set.
set -euo pipefail

RECIPE_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
# shellcheck disable=SC1091
. "$RECIPE_DIR/path.sh"

DATA_DIR="$RECIPE_DIR/data"
WAV_BASE="$DATA_DIR/wav"

run_prep() {
    local hf_split=$1 kaldi_name=$2 token=$3
    local out="$DATA_DIR/$kaldi_name"
    local wav="$WAV_BASE/$kaldi_name"
    if [[ -f "$out/wav.scp" ]]; then
        echo "  skip $kaldi_name (exists)"; return
    fi
    echo "  $hf_split -> $kaldi_name"
    python "$RECIPE_DIR/local/data_prep.py" \
        --hf_data_dir "$HF_DATA_DIR" \
        --split       "$hf_split" \
        --output_dir  "$out" \
        --wav_dir     "$wav" \
        --region_token "$token"
}

echo "=== Stage 1: Per-region data prep ==="
run_prep hidalgo-train              nahuatl_hidalgo_train              "<nah_hid>"
run_prep hidalgo-val                nahuatl_hidalgo_valid              "<nah_hid>"
run_prep hidalgo-test               nahuatl_hidalgo_test               "<nah_hid>"
run_prep orizaba-zongolica-train    nahuatl_orizaba_zongolica_train    "<nah_ozg>"
run_prep orizaba-zongolica-val      nahuatl_orizaba_zongolica_valid    "<nah_ozg>"
run_prep orizaba-zongolica-test     nahuatl_orizaba_zongolica_test     "<nah_ozg>"
run_prep zacatlan-tepetzintla-train nahuatl_zacatlan_tepetzintla_train "<nah_ztp>"
run_prep zacatlan-tepetzintla-val   nahuatl_zacatlan_tepetzintla_valid "<nah_ztp>"
run_prep zacatlan-tepetzintla-test  nahuatl_zacatlan_tepetzintla_test  "<nah_ztp>"

echo "=== Stage 2: Merge by set ==="
for setname in train valid test; do
    out="$DATA_DIR/nahuatl_${setname}"
    mkdir -p "$out"
    rm -f "$out/wav.scp" "$out/text" "$out/utt2spk"

    for region in hidalgo orizaba_zongolica zacatlan_tepetzintla; do
        src="$DATA_DIR/nahuatl_${region}_${setname}"
        cat "$src/wav.scp" >> "$out/wav.scp"
        cat "$src/text"    >> "$out/text"
        cat "$src/utt2spk" >> "$out/utt2spk"
    done

    sort -k1 -o "$out/wav.scp" "$out/wav.scp"
    sort -k1 -o "$out/text"    "$out/text"
    sort -k1 -o "$out/utt2spk" "$out/utt2spk"

    # Generate text.prev (<na> for all) and text.ctc (clean transcript)
    # NOTE: dotted filenames are the ESPnet s2t.sh convention for utt_extra_files;
    # the data names become text_prev / text_ctc (s2t.sh maps '.' -> '_').
    python3 -c "
import re, sys
tok = re.compile(r'<(nah_hid|nah_ozg|nah_ztp|asr|notimestamps)>\s*')
with open('$out/text') as f, \
     open('$out/text.prev', 'w') as fp, \
     open('$out/text.ctc', 'w') as fc:
    for line in f:
        uid, *rest = line.strip().split(None, 1)
        clean = tok.sub('', rest[0] if rest else '').strip()
        fp.write(f'{uid} <na>\n')
        fc.write(f'{uid} {clean}\n')
"

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
