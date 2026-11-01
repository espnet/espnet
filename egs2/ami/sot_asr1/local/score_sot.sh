#!/usr/bin/env bash
# Score one SOT decode directory: utterance-group cpWER + speaker-counting
# accuracy (local/evaluate_sot.py) and utterance-group DER (local/score_der.py,
# via SCTK md-eval.pl). All scorers use only dependencies already required by
# ESPnet; source path.sh first so md-eval.pl is on PATH.
#
# Usage:
#   local/score_sot.sh <decode_dir> <ref_dir> <out_dir> [cleaner] [collar]
#     decode_dir : dir with 1best_recog/{text,text_sot}
#     ref_dir    : data dir with the reference SOT `text`
#     out_dir    : where to write the scoring json files
set -euo pipefail

decode_dir=$1
ref_dir=$2
out_dir=$3
cleaner=${4:-whisper_en}
collar=${5:-0.25}

hyp_text="${decode_dir}/1best_recog/text"
hyp_text_sot="${decode_dir}/1best_recog/text_sot"
ref_text="${ref_dir}/text"

for f in "${hyp_text}" "${hyp_text_sot}" "${ref_text}"; do
    [ -f "${f}" ] || { echo "score_sot.sh: missing ${f}" >&2; exit 1; }
done

mkdir -p "${out_dir}"

echo "[score_sot.sh] cpWER + speaker counting -> ${out_dir}"
python local/evaluate_sot.py \
    --hyp_text "${hyp_text}" \
    --ref_text "${ref_text}" \
    --output_dir "${out_dir}" \
    --cleaner "${cleaner}"

echo "[score_sot.sh] DER (md-eval.pl, collar=${collar}) -> ${out_dir}"
python local/score_der.py \
    --hyp_text_sot "${hyp_text_sot}" \
    --ref_text "${ref_text}" \
    --output_dir "${out_dir}" \
    --collar "${collar}"

echo "[score_sot.sh] Done. Summary:"
python - "${out_dir}" <<'PY'
import json, os, sys
d = sys.argv[1]
cp = json.load(open(os.path.join(d, "cpwer.json")))
der = json.load(open(os.path.join(d, "der.json")))
spk = json.load(open(os.path.join(d, "speaker_count.json")))
print(f"  cpWER={cp['cpwer']:.2f}%  DER={der['der']:.2f}%  "
      f"spk-count-acc={spk['speaker_count_accuracy']:.2f}%")
PY
