#!/usr/bin/env bash
set -euo pipefail

test_sets=

. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 [--test_sets test] <s2t_exp>" >&2
    exit 1
fi

s2t_exp=$1

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -n "${test_sets}" ]; then
    hyp_files=()
    for dset in ${test_sets}; do
        while IFS= read -r -d '' hyp; do
            hyp_files+=("${hyp}")
        done < <(find "${s2t_exp}" -path "*/${dset}/text_nospecial" -print0)
    done
else
    hyp_files=()
    while IFS= read -r -d '' hyp; do
        hyp_files+=("${hyp}")
    done < <(find "${s2t_exp}" -name text_nospecial -print0)
fi

if [ ${#hyp_files[@]} -eq 0 ]; then
    log "No text_nospecial files found under ${s2t_exp}; skip ST BLEU scoring"
    exit 0
fi

for hyp in "${hyp_files[@]}"; do
    dir=$(dirname "${hyp}")
    dset=$(basename "${dir}")
    ref="dump/raw/${dset}/text"

    if [ ! -f "${ref}" ]; then
        log "Reference ${ref} not found for ${hyp}; skip"
        continue
    fi

    scoredir="${dir}/score_st_bleu"
    mkdir -p "${scoredir}"

    python - "${ref}" "${hyp}" "${scoredir}/ref.txt" "${scoredir}/hyp.txt" <<'PY'
import re
import sys


def read_kaldi_text(path, *, strip_prompt=False):
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt = parts[0]
            text = parts[1] if len(parts) == 2 else ""
            if strip_prompt:
                text = re.sub(r"^(?:<[^>]+>)+\s*", "", text).strip()
            data[utt] = text
    return data


ref = read_kaldi_text(sys.argv[1], strip_prompt=True)
hyp = read_kaldi_text(sys.argv[2])
keys = sorted(set(ref) & set(hyp))

missing_ref = len(set(hyp) - set(ref))
missing_hyp = len(set(ref) - set(hyp))
if missing_ref or missing_hyp:
    print(
        f"Aligned {len(keys)} utterances "
        f"({missing_ref} hyp-only, {missing_hyp} ref-only)",
        file=sys.stderr,
    )

with open(sys.argv[3], "w", encoding="utf-8") as f_ref, open(
    sys.argv[4], "w", encoding="utf-8"
) as f_hyp:
    for key in keys:
        f_ref.write(ref[key] + "\n")
        f_hyp.write(hyp[key] + "\n")
PY

    echo "Case-sensitive ST result (single-reference)" > "${scoredir}/result.txt"
    sacrebleu "${scoredir}/ref.txt" \
        -i "${scoredir}/hyp.txt" \
        -m bleu chrf ter \
        >> "${scoredir}/result.txt"

    log "Write ST BLEU/chrF/TER result in ${scoredir}/result.txt"
    cat "${scoredir}/result.txt"
done
